import base64
import os
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import copy
import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64

from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT
from qwen_vl.model.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGenerationWithGenerative,
)
from qwen_vl.data.utils import load_and_preprocess_images

try:
    # from qwen_vl_utils import process_vision_info
    from qwen_vl_utils import extract_vision_info
except ImportError:
    eval_logger.warning(
        "Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`"
    )


@register_model("vgllm")
class VGLLM(lmms):

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        use_flash_attention_2: Optional[bool] = False,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[
            float
        ] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[
            int
        ] = None,  # Only applicable if use_custom_video_loader is True
        max_length: Optional[int] = None,
        add_frame_index: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        self.add_frame_index = add_frame_index
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError(
                "max_image_size is only applicable if use_custom_video_loader is True"
            )

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        config = AutoConfig.from_pretrained(pretrained)
        # Force data-parallel style evaluation across processes; disable model-internal TP plans.
        if hasattr(config, "base_model_tp_plan"):
            config.base_model_tp_plan = None
        if hasattr(config, "base_model_pp_plan"):
            config.base_model_pp_plan = None
        if hasattr(config, "pretraining_tp"):
            config.pretraining_tp = 1

        if getattr(config, "use_generative_encoder", False):
            load_class = Qwen2_5_VLForConditionalGenerationWithGenerative
            eval_logger.info("Using Qwen2_5_VLForConditionalGenerationWithGenerative")
        elif getattr(config, "use_geometry_encoder", False) or getattr(
            config, "use_vggt_feature", False
        ):
            load_class = Qwen2_5_VLForConditionalGenerationWithVGGT
            eval_logger.info("Using Qwen2_5_VLForConditionalGenerationWithVGGT")
        else:
            load_class = Qwen2_5_VLForConditionalGeneration
            eval_logger.info("Using Qwen2_5_VLForConditionalGeneration")
        if use_flash_attention_2:
            self._model = load_class.from_pretrained(
                pretrained,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                attn_implementation="flash_attention_2",
            ).eval()
        else:
            self._model = load_class.from_pretrained(
                pretrained,
                config=config,
                torch_dtype="auto",
                device_map=self.device_map,
            ).eval()
        if hasattr(self._model, "_tp_plan"):
            self._model._tp_plan = None
        if hasattr(self._model, "_pp_plan"):
            self._model._pp_plan = None
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        self.processor = AutoProcessor.from_pretrained(
            pretrained,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            padding_side="left",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="left")

        if max_length is not None:
            eval_logger.warning(f"Setting max_length to {max_length}")
            setattr(self.processor.tokenizer, "model_max_length", max_length)
            setattr(self._tokenizer, "model_max_length", max_length)

        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # Keep per-rank model local and avoid `prepare_model`, which may trigger
            # DTensor/TP pathways depending on runtime/transformers versions.
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(
                    f"Using {accelerator.num_processes} devices with per-rank inference "
                    "(accelerator.prepare_model is skipped)."
                )
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1
            self._model = self.model.to("cuda").to(torch.bfloat16)

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(
            total=len(requests), disable=(self.rank != 0), desc="Model Responding"
        )
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator(
            [reg.args for reg in requests], _collate, grouping=True
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            docs = [
                self.task_dict[cur_task][cur_split][ids]
                for ids, cur_task, cur_split in zip(doc_id, task, split)
            ]
            visuals_per_sample = [
                visual_fn(doc) for visual_fn, doc in zip(doc_to_visual, docs)
            ]

            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(
                        f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}"
                    )

            messages = []
            sample_visual_meta = []
            for i, context in enumerate(contexts):

                message = [
                    {"role": "system", "content": "You are a helpful assistant."}
                ]
                visual_meta: Dict[str, Any] = {
                    "video_path": None,
                    "video_total_frames": None,
                    "sampled_frame_indices": [],
                    "num_sampled_frames": 0,
                }
                sample_visuals = visuals_per_sample[i]
                if not isinstance(sample_visuals, (list, tuple)):
                    sample_visuals = [sample_visuals]
                visual = None
                if len(sample_visuals) == 1:
                    visual = sample_visuals[0]
                elif len(sample_visuals) > 1:
                    visual = sample_visuals

                if visual is not None:
                    if isinstance(visual, str) and visual.endswith(
                        (".mp4", ".avi", ".mov")
                    ):  # Video file
                        vr = decord.VideoReader(visual)
                        image_num = len(vr)
                        # sample max_num_frames frame indices from the video
                        if image_num < self.max_num_frames:
                            frame_indices = np.arange(image_num)
                        else:
                            frame_indices = np.linspace(
                                0, image_num - 1, self.max_num_frames
                            ).astype(int)
                        # read the frames
                        frames = [vr[i].asnumpy() for i in frame_indices]
                        visual_content = []
                        visual_meta = {
                            "video_path": visual,
                            "video_total_frames": int(image_num),
                            "sampled_frame_indices": [
                                int(index) for index in frame_indices.tolist()
                            ],
                            "num_sampled_frames": int(len(frame_indices)),
                        }
                        for frame in frames:
                            image = Image.fromarray(frame).convert("RGB")
                            visual_content.append({"type": "image", "image": image})
                        message.append(
                            {
                                "role": "user",
                                "content": visual_content
                                + [{"type": "text", "text": context}],
                            }
                        )

                    elif isinstance(visual, Image.Image):  # Single image
                        base64_image = visual.convert("RGB")
                        buffer = BytesIO()
                        base64_image.save(buffer, format="JPEG")
                        base64_bytes = base64.b64encode(buffer.getvalue())
                        base64_string = base64_bytes.decode("utf-8")
                        message.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "image": f"data:image/jpeg;base64,{base64_string}",
                                    },
                                    {"type": "text", "text": context},
                                ],
                            }
                        )
                    elif isinstance(visual, (list, tuple)) and all(
                        isinstance(v, Image.Image) for v in visual
                    ):  # Multiple images
                        image_content = []
                        image_count = 0
                        for v in visual:
                            base64_image = v.convert("RGB")
                            buffer = BytesIO()
                            base64_image.save(buffer, format="JPEG")
                            base64_bytes = base64.b64encode(buffer.getvalue())
                            base64_string = base64_bytes.decode("utf-8")
                            if self.add_frame_index:
                                image_content.append(
                                    {
                                        "type": "text",
                                        "text": "Frame-{}: ".format(image_count),
                                    }
                                )
                            image_content.append(
                                {
                                    "type": "image",
                                    "image": f"data:image/jpeg;base64,{base64_string}",
                                }
                            )
                            image_count += 1
                        message.append(
                            {
                                "role": "user",
                                "content": image_content
                                + [{"type": "text", "text": context}],
                            }
                        )
                    else:
                        message.append(
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": context}],
                            }
                        )
                else:
                    message.append(
                        {"role": "user", "content": [{"type": "text", "text": context}]}
                    )

                messages.append(message)
                sample_visual_meta.append(visual_meta)

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # image_inputs, video_inputs = process_vision_info(messages)

            geometry_encoder_inputs = []
            image_inputs = []
            patch_size = self.processor.image_processor.patch_size
            merge_size = self.processor.image_processor.merge_size
            for message in messages:
                vision_info = extract_vision_info(message)
                cur_geometry_encoder_inputs = []
                for ele in vision_info:
                    if "image" in ele:
                        image = ele["image"]
                        if isinstance(image, Image.Image):
                            pass
                        elif isinstance(image, str) and "base64," in image:
                            _, base64_data = image.split("base64,", 1)
                            data = base64.b64decode(base64_data)
                            # fix memory leak issue while using BytesIO
                            with BytesIO(data) as bio:
                                image = copy.deepcopy(Image.open(bio))
                        else:
                            raise NotImplementedError("Unsupported image type")

                    else:
                        raise NotImplementedError("Unsupported vision info type")

                    assert isinstance(
                        image, Image.Image
                    ), f"Unsupported image type: {type(image)}"
                    image = load_and_preprocess_images([image])[0]
                    _, height, width = image.shape
                    # merge_size = 2
                    if (width // patch_size) % merge_size > 0:
                        width = width - (width // patch_size) % merge_size * patch_size
                    if (height // patch_size) % merge_size > 0:
                        height = (
                            height - (height // patch_size) % merge_size * patch_size
                        )
                    image = image[:, :height, :width]
                    # Save the 28-px-grid-aligned image so the generative
                    # encoder sees the same spatial content as the Qwen encoder.
                    cur_geometry_encoder_inputs.append(copy.deepcopy(image))
                    image_inputs.append(image)

                geometry_encoder_inputs.append(torch.stack(cur_geometry_encoder_inputs))
            inputs = self.processor(
                text=text,
                images=image_inputs,
                videos=None,
                padding=True,
                return_tensors="pt",
                do_rescale=False,
            )
            device = "cuda" if self.device_map == "auto" else self.device
            if getattr(self.model.config, "use_generative_encoder", False):
                inputs["geometry_encoder_inputs"] = [
                    feat.to(device) for feat in geometry_encoder_inputs
                ]
            elif getattr(self.model.config, "use_geometry_encoder", False) or getattr(
                self.model.config, "use_vggt_feature", False
            ):
                inputs["geometry_encoder_inputs"] = [
                    feat.to(device) for feat in geometry_encoder_inputs
                ]
            inputs = inputs.to(device)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 4096
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            pad_token_id = self.tokenizer.pad_token_id

            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, cont)
            ]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for i, ans in enumerate(answers):
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), ans
                )
                pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
