"""
Lightweight SVD feature extractor.
Only keeps the minimum code paths needed for UNet feature extraction.
"""

from typing import Callable, Optional, Union, List, Dict

import torch
import PIL
from diffusers import StableVideoDiffusionPipeline
from transformers import CLIPVisionModelWithProjection

from .utils import (
    randn_tensor,
    retrieve_timesteps,
    _resize_with_antialiasing,
    _append_dims,
    customunet,
    handle_memory_attention,
)
from ..common import disable_hf_zero3_init


class SVDVisionBackbone:
    def __init__(
        self,
        vision_backbone_id: str,
        image_resize_strategy: str,
        default_image_size: int = 224,
        height: Optional[int] = 448,
        width: Optional[int] = 448,
        num_frames: Optional[int] = 8,
        svd_model_path: str = "data/models/stable-video-diffusion-img2vid",
        clip_model_path: str = "ckpts/CLIP-ViT-H-14-laion2B-s32B-b79K",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.vision_backbone_id = vision_backbone_id
        self.image_resize_strategy = image_resize_strategy
        self.default_image_size = default_image_size
        self.svd_model_path = svd_model_path
        self.clip_model_path = clip_model_path
        self.torch_dtype = torch_dtype
        self.height = height if height is not None else 448
        self.width = width if width is not None else 448
        self.num_frames = num_frames if num_frames is not None else 8

        self.pipeline, self.vae, self.unet = self._load_svd_models()
        self.image_encoder = self._load_image_encoder()
        # Use a standalone CLIP image encoder that is loaded outside HF ZeRO-3 init.
        self.pipeline.image_encoder = self.image_encoder
        self.vae_scale_factor = self.pipeline.vae_scale_factor

        handle_memory_attention(
            enable_xformers_memory_efficient_attention=False,
            enable_torch_2_attn=True,
            unet=self.unet,
        )

        self.to(self.torch_dtype)
        if self.vae is not None:
            self.vae.config.force_upcast = False

    def to(self, dtype: torch.dtype) -> None:
        self.pipeline.to(dtype=dtype)
        if self.vae is not None:
            self.vae.to(dtype=dtype)
        self.unet.to(dtype=dtype)
        self.image_encoder.to(dtype=dtype)

    def eval(self) -> "SVDVisionBackbone":
        if hasattr(self.pipeline, "eval"):
            self.pipeline.eval()
        if self.vae is not None:
            self.vae.eval()
        self.unet.eval()
        self.image_encoder.eval()
        return self

    def _load_svd_models(self):
        with disable_hf_zero3_init():
            pipeline = StableVideoDiffusionPipeline.from_pretrained(
                self.svd_model_path,
                torch_dtype=self.torch_dtype,
                variant="fp16",
                low_cpu_mem_usage=False,
                device_map=None,
            )
        config = pipeline.unet.config
        state_dict = pipeline.unet.state_dict()
        new_unet = customunet(config)
        new_unet.load_state_dict(state_dict)
        new_unet.to(dtype=self.torch_dtype)
        pipeline.unet = new_unet
        if pipeline.vae is not None:
            pipeline.vae.to(dtype=self.torch_dtype)
        return pipeline, pipeline.vae, pipeline.unet

    def _load_image_encoder(self):
        with disable_hf_zero3_init():
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                self.clip_model_path,
                low_cpu_mem_usage=False,
            )
        image_encoder.to(dtype=self.torch_dtype)
        image_encoder.requires_grad_(False)
        return image_encoder

    def _ensure_valid_image_encoder(self, device: torch.device, dtype: torch.dtype) -> None:
        patch_weight = self.pipeline.image_encoder.vision_model.embeddings.patch_embedding.weight
        if patch_weight.ndim >= 3:
            return

        # Recover from invalid ZeRO-3-partitioned tensor by replacing with standalone encoder.
        self.pipeline.image_encoder = self.image_encoder
        self.pipeline.image_encoder.to(device=device, dtype=dtype)
        patch_weight = self.pipeline.image_encoder.vision_model.embeddings.patch_embedding.weight
        if patch_weight.ndim < 3:
            raise RuntimeError(
                f"Invalid SVD image encoder patch conv weight shape: {tuple(patch_weight.shape)}. "
                "Expected >=3D tensor."
            )

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.unet.to(pixel_values.device, dtype=self.torch_dtype)
        if self.vae is not None:
            self.vae.to(pixel_values.device, dtype=self.torch_dtype)
        self.image_encoder.to(pixel_values.device, dtype=self.torch_dtype)

        is_multiframe = pixel_values.shape[1] > 1
        conditioning_image = pixel_values[:, 0]
        if is_multiframe:
            latents = self.ddim_one_step(
                image=conditioning_image,
                pipeline=self.pipeline,
                vae=self.vae,
                unet=self.unet,
                image_encoder=self.image_encoder,
                height=self.height,
                width=self.width,
                num_frames=self.num_frames,
                output_type="unet_latent",
                all_frames_pixels=pixel_values,
            )
        else:
            latents = self.ddim_one_step(
                image=conditioning_image,
                pipeline=self.pipeline,
                vae=self.vae,
                unet=self.unet,
                image_encoder=self.image_encoder,
                height=self.height,
                width=self.width,
                num_frames=self.num_frames,
                output_type="unet_latent",
            )
        latents_flat = latents.permute(0, 1, 3, 4, 2).reshape(latents.size(0), -1, latents.size(2))
        return latents_flat

    @torch.no_grad()
    def ddim_one_step(self, image, pipeline, vae, unet, image_encoder,
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = 8,
        num_inference_steps: int = 1, #1000
        sigmas: Optional[List[float]] = None,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        all_frames_pixels: Optional[torch.Tensor] = None,
    ):
        """
        Single model forward pass to extract features from one forward call.
        noise_shape = [args.bs, channels, n_frames, h, w]
        image [b,c,h,w]
        video [b,c,t,h,w]
        return:
            cognition_features: [B, T, D]
        """
        pipeline.vae.eval()
        image_encoder.eval()
        device = unet.device
        dtype = self.torch_dtype

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        num_frames = num_frames if num_frames is not None else unet.config.num_frames

        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        pipeline.check_inputs(image, height, width)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]

        pipeline._guidance_scale = max_guidance_scale
        do_classifier_free_guidance = True

        # 3. Encode input image
        def tensor_to_PIL(image):
            """Convert a PyTorch tensor batch to a list of PIL images."""
            from PIL import Image
            import numpy as np
            if not isinstance(image, torch.Tensor) or image.dim() != 4:
                raise TypeError("Input must be a 4D PyTorch tensor (B, C, H, W)")
            image = image.clamp(0, 1) * 255
            numpy_array = image.permute(0, 2, 3, 1).cpu().byte().numpy()
            pil_images = [Image.fromarray(img) for img in numpy_array]
            return pil_images

        image = tensor_to_PIL(image) if isinstance(image, torch.Tensor) else image

        def numpy_batch_to_pt(numpy_batch):
            """Convert a numpy batch array (B, H, W, C) to a PyTorch tensor (B, C, H, W)."""
            return torch.tensor(numpy_batch, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0

        if isinstance(image[0], PIL.Image.Image):
            # overwatch.info(f"Input is a list of PIL images, converting to tensors for CLIP Processor")
            pil_images = image.copy()
            import numpy as np
            numpy_arrays = [np.array(im) for im in pil_images]
            image_batch_np = np.stack(numpy_arrays, axis=0)
            image_batch_tensor = numpy_batch_to_pt(image_batch_np)
            image_batch_tensor = image_batch_tensor * 2.0 - 1.0
            image_batch_tensor = _resize_with_antialiasing(image_batch_tensor, (224, 224))
            image = (image_batch_tensor + 1.0) / 2.0
        else:
            pil_images = image

        pipeline.image_encoder.to(dtype=dtype, device=device)
        self._ensure_valid_image_encoder(device=device, dtype=dtype)
        # NOTE: need to hack numpy to support bf16 "pt": lambda obj: obj.detach().cpu().float().numpy().astype(ml_dtypes.bfloat16),
        # In site-packages/transformers/utils/generic.py
        # https://github.com/huggingface/diffusers/issues/7598
        # workaround: https://github.com/pytorch/pytorch/issues/109873#issuecomment-2019226035
        image_embeddings = pipeline._encode_image(image, device, num_videos_per_prompt, do_classifier_free_guidance)
        # image_embeddings.to(dtype=dtype)
        # NOTE: Stable Video Diffusion was conditioned on fps - 1, which is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        image = pipeline.video_processor.preprocess(pil_images, height=height, width=width).to(device=device, dtype=dtype)
        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
        image = image + noise_aug_strength * noise

        # needs_upcasting = False #pipeline.vae.dtype == torch.float16 and pipeline.vae.config.force_upcast
        # if needs_upcasting:
        #     vae.to(dtype=torch.float32)

        pipeline.vae.to(dtype=dtype, device=device)
        image_latents = pipeline._encode_vae_image(
            image,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

        if all_frames_pixels is not None and all_frames_pixels.shape[1] > 0:
            num_input_frames = all_frames_pixels.shape[1]

            all_frames_pixels_flat = all_frames_pixels.flatten(0, 1)

            batch_pils = tensor_to_PIL(all_frames_pixels_flat) if isinstance(all_frames_pixels_flat, torch.Tensor) else all_frames_pixels_flat # input is 4 dim

            input_frames = pipeline.video_processor.preprocess(batch_pils, height=height, width=width).to(device=device, dtype=dtype)
            input_frames = input_frames + noise_aug_strength * randn_tensor(input_frames.shape, generator=generator, device=device, dtype=input_frames.dtype)

            input_frames_latents = pipeline._encode_vae_image(
                input_frames,
                device=device,
                num_videos_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
            ) # [B * num_frames, c, h, w]

            image_latents_with_frames = input_frames_latents.unflatten(0, (batch_size*2, num_input_frames))

            indices_to_replace = torch.linspace(0, num_frames - 1, steps=num_input_frames).round().long()
            image_latents[:, indices_to_replace] = image_latents_with_frames


        # 5. Get Added Time IDs
        added_time_ids = pipeline._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 6. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(pipeline.scheduler, num_inference_steps, device, None, sigmas)
        timesteps.to(device, dtype=dtype)

        # 7. Prepare latent variables
        num_channels_latents = pipeline.unet.config.in_channels
        latents = pipeline.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 8. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        pipeline._guidance_scale = guidance_scale

        # 9. Denoising loop
        #num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        pipeline._num_timesteps = len(timesteps)

        for i, t in enumerate(timesteps[:]):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents # [bsz*2,frame,4,32,32] Doubling bsz for guidance
            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

            latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
            latent_model_input.to(dtype=latents.dtype)
            # predict the noise residual
            # latent_model_input = latent_model_input.to(pipeline.unet.dtype)
            # image_embeddings = image_embeddings.to(pipeline.unet.dtype)

            if output_type == "unet_latent":
                return_down_features = True  # We want the down features for unet_latent output
            else:
                return_down_features = False

            t.to(dtype=latent_model_input.dtype, device=latent_model_input.device)

            full_noise_pred, down_noise_pred = self.unet( #use custom unet forward
                latent_model_input,
                t,
                encoder_hidden_states=image_embeddings,
                added_time_ids=added_time_ids,
                return_dict=False,
                return_down_features=return_down_features,  # We want the down features
            )

            noise_pred = full_noise_pred

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + pipeline.guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            if output_type != "unet_latent":
                latents = pipeline.scheduler.step(noise_pred, t, latents).prev_sample

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)

            # if XLA_AVAILABLE:
            #         xm.mark_step()

        if output_type == "unet_latent":
            noise_pred_uncond, noise_pred_cond = down_noise_pred.chunk(2)  # [2 * b, t, c, h, w]
            frames = noise_pred_cond.to(dtype=dtype)
            pipeline.maybe_free_model_hooks()
            return frames
        elif not output_type == "latent":
            original_vae_dtype = pipeline.vae.dtype
            pipeline.vae.to(dtype=torch.float32)
            latents = latents.to(torch.float32)
            with torch.no_grad():
                frames = pipeline.decode_latents(latents, num_frames, decode_chunk_size)
                frames = pipeline.video_processor.postprocess_video(video=frames, output_type=output_type)
        else:
            frames = latents
            frames = frames.to(torch.float32)
        pipeline.maybe_free_model_hooks()
        return frames
