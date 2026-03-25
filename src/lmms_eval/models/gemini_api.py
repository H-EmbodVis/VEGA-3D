import io
import json
import os
import time
from typing import List, Tuple

from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

import hashlib

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmBlockThreshold, HarmCategory
    from google.api_core.exceptions import ResourceExhausted

    NUM_SECONDS_TO_SLEEP = os.getenv("GEMINI_INTV_AFTER_FAILED", 10)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)
    genai.configure(api_key=GOOGLE_API_KEY)

except Exception as e:
    eval_logger.error(f"Error importing generativeai: {str(e)}")
    genai = None

@register_model("gemini_api")
class GeminiAPI(lmms):
    def __init__(
        self,
        model_version: str = "gemini-1.5-pro",
        modality: str = "image",
        timeout: int = 120,
        continual_mode: bool = False,
        response_persistent_folder: str = None,  # We will cache the Gemini API response in this path and use it for future requests
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.timeout = timeout
        self.generation_config = {'temperature': 0, 'top_p': 1, 'top_k': 1, 'max_output_tokens': 8192}
        self.model = genai.GenerativeModel(model_version, generation_config=self.generation_config)
        self.continual_mode = continual_mode
        if self.continual_mode and response_persistent_folder is None:
            raise ValueError("Continual mode requires a persistent path for the response. We will cache the Gemini API response in this path and use it for future requests. Please provide a valid path.")
        self.response_persistent_folder = response_persistent_folder
        if self.continual_mode:
            if not os.path.exists(self.response_persistent_folder):
                os.makedirs(self.response_persistent_folder)
            self.response_persistent_file = os.path.join(self.response_persistent_folder, f"{self.model_version}_response.json")
            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
            else:
                self.response_cache = {}
                self.cache_mode = "start"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert self.continual_mode is False, "Continual mode is not supported with distributed inference."
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device

        self.modality = modality
        
    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def get_image_size(self, image):
        # Create a BytesIO object to store the image bytes
        img_byte_array = io.BytesIO()

        # Save the image to the BytesIO object
        image.save(img_byte_array, format="PNG")

        # Get the size of the BytesIO object
        img_size = img_byte_array.tell()

        return img_size

    def encode_video(self, video_path):
        uploaded_obj = genai.upload_file(path=video_path)
        time.sleep(5)
        return uploaded_obj

    def encode_image(self, image_path):
        uploaded_obj = genai.upload_file(path=image_path)
        time.sleep(5)
        return uploaded_obj

    def convert_video(self, images):
        for idx, img in enumerate(images):
            if self.modality == "video" and isinstance(img, str):
                try:
                    images[idx] = self.encode_video(img)
                except Exception as e:
                    eval_logger.error(f"Error converting video: {str(e)}")
            elif self.modality == 'image' and isinstance(img, str):
                try:
                    images[idx] = self.encode_image(img)
                except Exception as e:
                    eval_logger.error(f"Error converting image: {str(e)}")
        return images

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if self.continual_mode is True and self.cache_mode == "resume":
                doc_uuid = get_uuid(task, split, doc_id)
                if doc_uuid in self.response_cache:
                    content = self.response_cache[doc_uuid]
                    if content:
                        res.append(content)
                        pbar.update(1)
                        continue

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            if visuals != [None]:
                visuals = self.flatten(visuals)
                visuals = self.convert_video(visuals)

            if visuals != [None]: # always follow gemini's suggestion that take video first
                chat_session = self.model.start_chat(
                    history=[
                        {
                            "role": "user",
                            "parts": [visuals[0]]
                        },
                    ])
            else:
                chat_session = self.model.start_chat()

            for attempt in range(5):
                try:
                    content = chat_session.send_message(contexts)
                    content = content.text
                    break
                except Exception as e:
                    eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}")
                    if isinstance(e, ValueError):
                        try:
                            eval_logger.info(f"Prompt feed_back: {content.prompt_feedback}")
                            content = ""
                            break
                        except Exception:
                            pass
                    if attempt < 5 - 1:  # If we have retries left, sleep and then continue to next attempt
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:  # If this was the last attempt, log and return empty
                        eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}")
                        content = ""
                        if isinstance(e, ResourceExhausted):
                            eval_logger.error("Quota exceed!!!")
            res.append(content)
            pbar.update(1)

            if self.continual_mode is True:  # Cache the response
                doc_uuid = get_uuid(task, split, doc_id)
                self.response_cache[doc_uuid] = content
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "Gemini API not support"
