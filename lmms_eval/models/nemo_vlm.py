import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL
import torch
from decord import VideoReader, cpu
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from transformers import AutoProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from megatron.core.inference.common_inference_params import CommonInferenceParams

from nemo import lightning as nl
from nemo.collections import vlm
from nemo.collections.vlm.mllama.model.utils import create_vision_mask_tensor
from nemo.collections.vlm.inference import generate as vlm_generate
from nemo.collections.vlm.inference import setup_inference_wrapper

warnings.filterwarnings("ignore")

DEFAULT_IMAGE_TOKEN = "<|image|>"

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

@register_model("nemo_vlm")
class NeMoVLM(lmms):
    def __init__(
        self,
        pretrained: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
        load_from_hf: Optional[bool] = False,
        dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
        tp_size: Optional[int] = 1,
        batch_size: int = 1,
        max_frames_num: Optional[int] = 32,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        strategy = nl.MegatronStrategy(
            tensor_model_parallel_size=tp_size,
            ckpt_load_optimizer=False,
            ckpt_save_optimizer=False,
        )
        trainer = nl.Trainer(
            devices=tp_size,
            max_steps=1000,
            accelerator="gpu",
            strategy=strategy,
            plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
            val_check_interval=1000,
            limit_val_batches=50,
        )

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer

        fabric = trainer.to_fabric()

        if load_from_hf:
            self.model = fabric.import_model(f"hf://{pretrained}", vlm.MLlamaModel)
        else:
            self.model = vlm.MLlamaModel(vlm.MLlamaConfig11BInstruct(), tokenizer=self.tokenizer)
            self.model = fabric.load_model(pretrained, self.model)

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        self.model = self.model.module.cuda()
        self.model.eval()
        self.model = self.model.to(dtype)

        self.max_frames_num = max_frames_num
        self._rank = torch.distributed.get_rank()

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for context, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            messages = [{"role": "user", "content": []}]
            images = []
            # Apply chat template
            for visual in visuals:
                if isinstance(visual, str):
                    frames = self.load_video(visual, self.max_frames_num)
                    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
                    images.extend([to_pil_image(frame) for frame in frames])
                elif isinstance(visual, PIL.Image.Image):
                    images.append(visual)

            for _ in range(len(images)):
                messages[-1]["content"].append({"type": "image"})
            messages[-1]["content"].append({"type": "text", "text": context})
            messages.append({"role": "assistant", "content": continuation})

            prompt = self.processor.apply_chat_template(messages[:-1], add_generation_prompt=True)
            prompt_and_continuation = self.processor.apply_chat_template(messages, add_generation_prompt=False)

            batch = self.processor(images, prompt_and_continuation, add_special_tokens=False, return_tensors="pt")

            input_ids = batch["input_ids"].cuda(non_blocking=True)
            position_ids = (
                torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)
            )
            num_tiles = self.processor.image_processor.preprocess(images, return_tensors='pt')["num_tiles"]
            batch_masks = create_vision_mask_tensor(input_ids[0])

            labels = input_ids.clone()
            contxt_id = self.processor(text=prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
            labels[:, : contxt_id.shape[1]] = -100

            with torch.no_grad():
                outputs = self.model(
                    batch_images=batch["pixel_values"].cuda(non_blocking=True),
                    batch_masks=[batch_masks],
                    num_chunks=torch.tensor(num_tiles),
                    aspect_ratio_ids=batch["aspect_ratio_ids"].cuda(non_blocking=True),
                    tokens=input_ids,
                    position_ids=position_ids,
                    labels=labels
                )
            loss = outputs["loss"]
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = batch["input_ids"][:, contxt_id.shape[1] :]  # [1, seq]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : batch["input_ids"].shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def load_video(self, video_path, max_frames_num):
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames  # (frames, height, width, channels)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        model_wrapper = setup_inference_wrapper(self.model, self.processor.tokenizer)

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)

            messages = [{"role": "user", "content": []}]
            images = []

            for visual in visuals:
                if isinstance(visual, str):
                    frames = self.load_video(visual, self.max_frames_num)
                    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
                    images.extend([to_pil_image(frame) for frame in frames])
                elif isinstance(visual, PIL.Image.Image):
                    images.append(visual)

            for _ in range(len(images)):
                messages[-1]["content"].append({"type": "image"})
            messages[-1]["content"].append({"type": "text", "text": contexts})
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 1.0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = 0.0
            if "top_k" not in gen_kwargs:
                gen_kwargs["top_k"] = 0
            if "do_sample" in gen_kwargs and not gen_kwargs["do_sample"]:
                gen_kwargs["top_k"] = 1
                gen_kwargs["top_p"] = 0.0

            prompts = [prompt]
            image_list = [images]
            params = CommonInferenceParams(
                top_k=gen_kwargs["top_k"],
                top_p=gen_kwargs["top_p"],
                temperature=gen_kwargs["temperature"],
                num_tokens_to_generate=gen_kwargs["max_new_tokens"],
            )

            result = vlm_generate(
                model_wrapper,
                self.processor.tokenizer,
                self.processor.image_processor,
                prompts,
                image_list,
                inference_params=params,
            )
            generated_texts = list(result)[0].generated_text
            res.append(generated_texts)

            pbar.update(1)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for NeMoVLM")
