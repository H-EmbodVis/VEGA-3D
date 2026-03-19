#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import math
import re
import time
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector
from .multimodal_generative_encoder.builder import build_generative_vision_tower

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random
from torch_scatter import scatter_mean
from .feature_fusion import FeatureFusionModule, FeatureFusionConfig

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))
        
        if hasattr(self.config, 'world_position_embedding_type'):
            from llava.model.position_encoding import PositionEmbeddingSine3D, PositionEmbeddingMLP

            if "sample9" in self.config.world_position_embedding_type:
                n_points = 9
            elif "sample5" in self.config.world_position_embedding_type:
                n_points = 5
            elif "minmax" in self.config.world_position_embedding_type:
                n_points = 2
            else:
                n_points = 1
        
            if "mlp" in self.config.world_position_embedding_type:
                self.world_position_embedding = PositionEmbeddingMLP(config.hidden_size, n_points=n_points)
            elif "sin3d" in self.config.world_position_embedding_type:
                self.world_position_embedding = PositionEmbeddingSine3D(config.hidden_size, n_points=n_points)
            # elif "slp" in self.config.world_position_embedding_type:
            #     self.world_position_embedding = PositionEmbeddingSine3DMLP(config.hidden_size, n_points=n_points)
        self.feature_fusion = None
        if getattr(self.config, "use_generative_feature", False):
            generative_feat_dim_dim = getattr(self.config, "generative_feat_dim", 1280)
            generative_proj_type = getattr(self.config, "generative_projector_type", "mlp2x_gelu")
            mm_projector_type = getattr(self.config, "mm_projector_type", "mlp2x_gelu")
            self.generative_feature_source = getattr(self.config, "generative_feature_source", "offline")
            # NOTE: keep online generative encoder lazy-initialized in forward.
            # Building WAN modules in __init__ under ZeRO-3 can create partitioned
            # zero-sized parameters and break checkpoint loading.
            self.generative_vision_tower = None
            # Runtime-only holder. Keep out of nn.Module registration to avoid
            # ZeRO-3 partition metadata issues for dynamically created parameters.
            self.__dict__["_generative_vision_tower_runtime"] = None

            self.config.mm_projector_type = generative_proj_type
            self.generative_projector = build_vision_projector(
                self.config,
                vision_cfg=self.vision_tower.config,
                in_dim=generative_feat_dim_dim,
            )
            self.config.mm_projector_type = mm_projector_type
            if getattr(self.config, "use_feat_fusion", False):
                fusion_config = FeatureFusionConfig(
                    fusion_method=getattr(self.config, "feature_fusion_method", "add"),
                    hidden_size=config.hidden_size,
                    num_heads=getattr(self.config, "fusion_attention_heads", 8),
                    dropout=getattr(self.config, "fusion_dropout", 0.1),
                    num_layers=getattr(self.config, "fusion_num_layers", 1),
                )
                self.feature_fusion = FeatureFusionModule(fusion_config)
            else:
                self.feature_fusion = None
        else:
            self.generative_projector = None
            self.generative_vision_tower = None

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        
        if not hasattr(self.config, 'add_faster_video'):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def _get_runtime_generative_vision_tower(self):
        model = self.get_model()
        runtime_tower = model.__dict__.get("_generative_vision_tower_runtime", None)
        if runtime_tower is None:
            runtime_tower = build_generative_vision_tower(self.config)
            runtime_tower.eval()
            for p in runtime_tower.parameters():
                p.requires_grad = False
            model.__dict__["_generative_vision_tower_runtime"] = runtime_tower
        return runtime_tower

    def _get_generative_cache(self):
        model = self.get_model()
        cache = model.__dict__.get("_generative_feature_cache", None)
        if cache is None:
            cache = OrderedDict()
            model.__dict__["_generative_feature_cache"] = cache
        return cache

    def _profile_generative(self, **kwargs):
        if not getattr(self.config, "generative_profile", False):
            return
        model = self.get_model()
        state = model.__dict__.get("_generative_profile_state", None)
        if state is None:
            state = {"count": 0, "tower": 0.0, "projector": 0.0, "cache_hits": 0, "cache_queries": 0}
            model.__dict__["_generative_profile_state"] = state
        for k, v in kwargs.items():
            if k in state:
                state[k] += v
        state["count"] += 1
        if state["count"] % 10 == 0:
            avg_tower = state["tower"] / state["count"]
            avg_proj = state["projector"] / state["count"]
            hit_rate = (state["cache_hits"] / max(1, state["cache_queries"])) * 100.0
            rank0_print(
                f"[generative_profile] steps={state['count']} avg_tower={avg_tower:.3f}s "
                f"avg_projector={avg_proj:.3f}s cache_hit={hit_rate:.1f}%"
            )

    @staticmethod
    def _stable_unique_token_ids(token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.numel() == 0:
            return token_ids
        seen = set()
        out = []
        for token_id in token_ids.tolist():
            token_id = int(token_id)
            if token_id not in seen:
                seen.add(token_id)
                out.append(token_id)
        return torch.tensor(out, device=token_ids.device, dtype=token_ids.dtype)

    def _build_instruction_context_vectors(self, input_ids, labels, attention_mask):
        """
        Build per-sample instruction context vectors for instruction-aware fusion.
        - Pure human: prefer labels==IGNORE_INDEX tokens.
        - De-template: drop common template token ids.
        - Keep relation words: preserve and upweight relation token ids.
        """
        if input_ids is None:
            return None

        embed_tokens = self.get_model().embed_tokens
        embed_device = embed_tokens.weight.device
        embed_dtype = embed_tokens.weight.dtype

        if attention_mask is None:
            valid_masks = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            valid_masks = attention_mask.bool()

        relation_ids = getattr(self.config, "instruction_relation_token_ids", [])
        template_ids = getattr(self.config, "instruction_template_token_ids", [])
        relation_boost = float(getattr(self.config, "instruction_relation_boost", 2.0))
        tail_keep_tokens = int(getattr(self.config, "instruction_tail_keep_tokens", 96))

        relation_tensor = None
        template_tensor = None
        if isinstance(relation_ids, (list, tuple)) and len(relation_ids) > 0:
            relation_tensor = torch.tensor(sorted(set(int(x) for x in relation_ids if int(x) >= 0)), dtype=input_ids.dtype, device=input_ids.device)
        if isinstance(template_ids, (list, tuple)) and len(template_ids) > 0:
            template_tensor = torch.tensor(sorted(set(int(x) for x in template_ids if int(x) >= 0)), dtype=input_ids.dtype, device=input_ids.device)

        contexts = []
        batch_size = input_ids.shape[0]
        for batch_i in range(batch_size):
            ids = input_ids[batch_i][valid_masks[batch_i]]
            if ids.numel() == 0:
                contexts.append(None)
                continue

            if labels is not None:
                cur_labels = labels[batch_i][valid_masks[batch_i]]
                human_mask = cur_labels.eq(IGNORE_INDEX)
            else:
                human_mask = torch.ones_like(ids, dtype=torch.bool)

            # Keep tokens after the first image token to avoid system/template prefix.
            image_pos = (ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False)
            if image_pos.numel() > 0:
                first_image_idx = int(image_pos[0].item())
                pos = torch.arange(ids.shape[0], device=ids.device)
                human_mask = human_mask & (pos > first_image_idx)

            token_ids = ids[human_mask]
            token_ids = token_ids[token_ids >= 0]

            if token_ids.numel() == 0:
                # Fallback for inference-style inputs without labels.
                token_ids = ids[ids >= 0]
            if token_ids.numel() == 0:
                contexts.append(None)
                continue

            # Remove prompt-template tokens but keep relation words.
            if template_tensor is not None and token_ids.numel() > 0:
                template_mask = torch.isin(token_ids, template_tensor)
                if relation_tensor is not None:
                    relation_mask = torch.isin(token_ids, relation_tensor)
                    template_mask = template_mask & (~relation_mask)
                token_ids = token_ids[~template_mask]
                if token_ids.numel() == 0:
                    token_ids = ids[ids >= 0]
                    if token_ids.numel() == 0:
                        contexts.append(None)
                        continue

            # Keep the tail to de-emphasize fixed prefixes while preserving relation words.
            if tail_keep_tokens > 0 and token_ids.numel() > tail_keep_tokens:
                tail_tokens = token_ids[-tail_keep_tokens:]
            else:
                tail_tokens = token_ids

            if relation_tensor is not None:
                relation_tokens = token_ids[torch.isin(token_ids, relation_tensor)]
                if relation_tokens.numel() > 0:
                    token_ids = self._stable_unique_token_ids(torch.cat([tail_tokens, relation_tokens], dim=0))
                else:
                    token_ids = tail_tokens
            else:
                token_ids = tail_tokens

            if token_ids.numel() == 0:
                contexts.append(None)
                continue

            token_ids = token_ids.to(embed_device)
            token_embeds = embed_tokens(token_ids).to(dtype=embed_dtype)

            weights = torch.ones(token_embeds.shape[0], device=embed_device, dtype=token_embeds.dtype)
            if relation_tensor is not None and relation_boost > 1.0:
                relation_on_device = relation_tensor.to(embed_device)
                relation_mask = torch.isin(token_ids, relation_on_device)
                if relation_mask.any():
                    weights = torch.where(relation_mask, torch.full_like(weights, relation_boost), weights)

            context = (token_embeds * weights[:, None]).sum(dim=0) / weights.sum().clamp_min(1e-6)
            context = F.normalize(context.float(), dim=0, eps=1e-6).to(dtype=embed_dtype)
            contexts.append(context)

        return contexts

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature


    def average_coordinate_in_patch(self, world_coords, patch_size=27):

        V, H, W, D = world_coords.size() # D = 3

        world_coords = world_coords.view(V, H, W, D)[:, :-6, :-6, :]    # [32, 378, 378, 3]
        world_coords = world_coords.permute(0, 3, 1, 2)   # [V, D, 378, 378]
        world_coords_avg = torch.nn.functional.avg_pool2d(world_coords, kernel_size=patch_size, stride=patch_size)  # [32, 3, 14,  14]
        patch_num = world_coords_avg.shape[-1]
        world_coords_avg = world_coords_avg.permute(0, 2, 3, 1)     # [32, 14, 14, 3]

        return world_coords_avg

    def minmax_coordinate_in_patch(self, world_coords, patch_size=27):

        V, H, W, D = world_coords.size() # D = 3

        world_coords = world_coords.view(V, H, W, D)[:, :-6, :-6, :]    # [32, 378, 378, 3]
        world_coords = world_coords.permute(0, 3, 1, 2)   # [V, D, 378, 378]

        world_coords_max = torch.nn.functional.max_pool2d(world_coords, kernel_size=patch_size, stride=patch_size)  # [32, 3, 14,  14]
        world_coords_max = world_coords_max.permute(0, 2, 3, 1)     # [32, 14, 14, 3]

        world_coords_min = - torch.nn.functional.max_pool2d(-world_coords, kernel_size=patch_size, stride=patch_size)  # [32, 3, 14,  14]
        world_coords_min = world_coords_min.permute(0, 2, 3, 1)     # [32, 14, 14, 3]
        world_coords = torch.stack([world_coords_min, world_coords_max], dim=3) # [32, 14, 14, 2, 3]

        return world_coords
    
    def sample_n_points(self, world_coords, n_points=9):

        V, H, W, D = world_coords.size() # D = 3
        world_coords = world_coords.view(V, H, W, D)[:, :-6, :-6, :] 
        world_coords = world_coords.view(-1, 14, 27, 14, 27, 3).permute(0, 1, 3, 2, 4, 5)
        if n_points == 9:
            world_coords_sample = world_coords[:, :, :, 4::9, 4::9, :].reshape(V, 14, 14, 9, 3)
        elif n_points == 5:
            world_coords_sample = world_coords[:, :, :, 4::9, 4::9, :].reshape(V, 14, 14, 9, 3)
            world_coords_sample = world_coords_sample[:, :, :, 0::2, :].reshape(V, 14, 14, 5, 3)
        elif n_points == 1:
            world_coords_sample = world_coords[:, :, :, 4::9, 4::9, :].reshape(V, 14, 14, 9, 3)
            world_coords_sample = world_coords_sample[:, :, :, 4, :].reshape(V, 14, 14, 3)
        else:
            raise NotImplementedError
        
        return world_coords_sample

    def discrete_coords(self, world_coords, xyz_min):

        # V, H, W, D = world_coords.size() # D = 3
        # world_coords_discrete = (world_coords.view(-1, 3) - xyz_min.view(1, 3)) / self.config.voxel_size

        min_xyz_range = torch.tensor(self.config.min_xyz_range).to(world_coords.device)
        max_xyz_range = torch.tensor(self.config.max_xyz_range).to(world_coords.device)

        world_coords = torch.maximum(world_coords, min_xyz_range)
        world_coords = torch.minimum(world_coords, max_xyz_range)
        world_coords_discrete = (world_coords - min_xyz_range) / self.config.voxel_size
        world_coords_discrete = world_coords_discrete.round()

        return world_coords_discrete.detach()


    def encode_images(self, images, world_coords=None):
        image_features = self.get_model().get_vision_tower()(images)
        mm_projector = self.get_model().mm_projector
        # Keep projector input dtype aligned with projector params; avoid relying on autocast across towers.
        try:
            proj_param = next(mm_projector.parameters())
            if image_features.dtype != proj_param.dtype:
                image_features = image_features.to(dtype=proj_param.dtype)
        except StopIteration:
            pass
        image_features = mm_projector(image_features)
        return image_features


    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)
        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []
        all_faster_video_features = []
        cur_mm_spatial_pool_stride = self.config.mm_spatial_pool_stride

        for idx, feat in enumerate(per_videos_or_images_features):
            mm_projector = self.get_model().mm_projector
            try:
                proj_param = next(mm_projector.parameters())
                if feat.dtype != proj_param.dtype:
                    feat = feat.to(dtype=proj_param.dtype)
            except StopIteration:
                pass
            feat = mm_projector(feat)
            faster_video_feature = 0
            slower_img_feat = 0
            if idx in video_idx_in_batch and cur_mm_spatial_pool_stride > 1:
                slower_img_feat = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
                if self.config.add_faster_video:
                    cur_mm_spatial_pool_stride = cur_mm_spatial_pool_stride * 2
                    faster_video_feature = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
            if slower_img_feat != 0:
                all_videos_or_images_features.append(slower_img_feat)
            else:
                all_videos_or_images_features.append(feat)
            all_faster_video_features.append(faster_video_feature)
        return all_videos_or_images_features,all_faster_video_features

    def add_token_per_grid(self, image_feature):
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]

        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        if getattr(self.config, "add_faster_video", False):
            # import pdb; pdb.set_trace()
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames,resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            # import pdb; pdb.set_trace()
            return image_feature
        # import pdb; pdb.set_trace()
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        return image_feature

    def add_token_per_frame(self, image_feature):
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature

    def prepare_inputs_labels_for_multimodal(
        self, 
        input_ids, 
        position_ids, 
        attention_mask, 
        past_key_values, 
        labels, 
        images, 
        modalities=["image"], 
        image_sizes=None, 
        video_dict=None,
        use_object_proposals: bool = False,
    ):
        object_boxes = None
        if use_object_proposals:
            object_boxes = video_dict["objects"][0]
            object_boxes_center = object_boxes[:, :3]
            object_features = []
            obj_num = len(object_boxes)

            object_patch = []
            # ignore the batch dimension here
            world_coords = video_dict["world_coords"][0]

            for l in range(obj_num):
                box = object_boxes[l]
                min_xyz = box[:3] - box[3:] / 2
                max_xyz = box[:3] + box[3:] / 2
                
                if "patch27" in self.config.object_feature_type:
                    world_coords_new = world_coords[:, :378, :378, :].reshape(-1, 14, 27, 14, 27, 3).transpose(2, 3).flatten(3, 4)  # [32, 14, 14, 27*27, 3]
                    cur_object_patch = torch.all((min_xyz <= world_coords_new) & (world_coords_new <= max_xyz), dim=-1)     # [32, 14, 14, 27*27]
                    cur_object_patch = cur_object_patch.sum(dim=3) >= int(27 * 27 * 0.25)
                    object_patch.append(cur_object_patch)
                elif "patch14" in self.config.object_feature_type:
                    world_coords_new = world_coords[:, :378, :378, :].reshape(-1, 27, 14, 27, 14, 3).transpose(2, 3).flatten(3, 4)  # [32, 14, 14, 27*27, 3]
                    cur_object_patch = torch.all((min_xyz <= world_coords_new) & (world_coords_new <= max_xyz), dim=-1)     # [32, 14, 14, 27*27]
                    cur_object_patch = cur_object_patch.sum(dim=3) >= int(14 * 14 * 0.5)
                    object_patch.append(cur_object_patch)
                else:
                    raise NotImplementedError
        
        
        use_mrope_position_embedding = False
        use_sin3d_pe = False
        use_mlp_pe = False
        if hasattr(self.config, 'world_position_embedding_type') and past_key_values is None:
            B = input_ids.shape[0]
            world_coords = video_dict['world_coords']
            xyz_min = world_coords.view(B, -1, 3).min(dim=1)[0]

            if len(video_dict['box_input']):
                box_input = video_dict['box_input']     # [1, 3]
            else:
                box_input = None

            n_points = 1
            if 'avg' in self.config.world_position_embedding_type:
                world_coords = [self.average_coordinate_in_patch(coords) for coords in world_coords]
            elif "sample9" in self.config.world_position_embedding_type:
                world_coords = [self.sample_n_points(coords, n_points=9) for coords in world_coords]
                n_points = 9
            elif "sample5" in self.config.world_position_embedding_type:
                world_coords = [self.sample_n_points(coords, n_points=5) for coords in world_coords]
                n_points = 5
            elif "sample1" in self.config.world_position_embedding_type:
                world_coords = [self.sample_n_points(coords, n_points=1) for coords in world_coords]
            elif "minmax" in self.config.world_position_embedding_type:
                world_coords = [self.minmax_coordinate_in_patch(coords) for coords in world_coords]
                n_points = 2

            if n_points > 1:
                if box_input is not None:
                    box_input = box_input[:, None, :].repeat(1, n_points, 1)
                if object_boxes is not None:
                    object_boxes_center = object_boxes_center[:, None, :].repeat(1, n_points, 1)

            if 'discrete' in self.config.world_position_embedding_type or use_mrope_position_embedding:
                world_coords_discrete = [self.discrete_coords(coords, xyz_min[i]) for i, coords in enumerate(world_coords)]
                if box_input is not None:
                    box_input = self.discrete_coords(box_input, None)
                if object_boxes is not None:
                    object_boxes_center = self.discrete_coords(object_boxes_center, None)

            if 'mrope' in self.config.world_position_embedding_type:
                use_mrope_position_embedding = True
            
            if "sin3d" in self.config.world_position_embedding_type:
                use_sin3d_pe = True
            
            if "mlp" in self.config.world_position_embedding_type:
                use_mlp_pe = True


        vision_tower = self.get_vision_tower()
        # rank_print(modalities)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None

        if isinstance(modalities, str):
            modalities = [modalities]

        instruction_contexts = None
        if (
            getattr(self.config, "use_generative_feature", False)
            and getattr(self.config, "use_feat_fusion", False)
            and getattr(self.config, "feature_fusion_method", "add") in {"instruction_token_gated", "context_gated"}
        ):
            instruction_contexts = self._build_instruction_context_vectors(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )

        # import pdb; pdb.set_trace()
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            encoded_image_features = self.encode_images(concat_images)
            encoded_image_features = torch.split(encoded_image_features, split_sizes) #[32, 729, 1152] (projector)-> [32, 729, 3584]
            generative_features_list = [None] * len(images_list)
            use_generative_feature = getattr(self.config, "use_generative_feature", False)
            use_feat_fusion = getattr(self.config, "use_feat_fusion", False)
            generative_feature_source = getattr(self.config, "generative_feature_source", "offline")
            
            if use_generative_feature:
                if generative_feature_source == "offline":
                    if video_dict is not None and ("generative_feature" in video_dict) and (video_dict["generative_feature"] is not None):
                        generative_features = video_dict["generative_feature"]  # [B, V, Cg, Hg, Wg]
                        B, V, Cg, Hg, Wg = generative_features.shape
                        generative_split_sizes = [V] * B
                        generative_features = generative_features.view(B * V, Cg, Hg, Wg)
                        generative_features = generative_features.to(concat_images.device)
                        generative_features = generative_features.view(B * V, Cg, Hg * Wg).permute(0, 2, 1).contiguous()
                        generative_features = self.get_model().generative_projector(generative_features)
                        offline_splits = list(torch.split(generative_features, generative_split_sizes))
                        for i in range(min(len(offline_splits), len(generative_features_list))):
                            generative_features_list[i] = offline_splits[i]
                elif generative_feature_source == "online":
                    runtime_tower = self._get_runtime_generative_vision_tower()
                    if len(video_idx_in_batch) > 0:
                        video_images = [images_list[idx] for idx in video_idx_in_batch]
                        video_split_sizes = [x.shape[0] for x in video_images]
                        video_ids = None
                        if video_dict is not None and "video_id" in video_dict and isinstance(video_dict["video_id"], list):
                            video_ids = video_dict["video_id"]

                        cache = self._get_generative_cache()
                        max_cache_items = int(getattr(self.config, "generative_cache_max_items", -1))
                        cache_enabled = max_cache_items != 0
                        cache_queries = 0
                        cache_hits = 0
                        online_pooled_per_video = [None] * len(video_images)

                        miss_images = []
                        miss_indices = []
                        miss_split_sizes = []
                        for local_i, vid_images in enumerate(video_images):
                            cache_key = None
                            batch_i = video_idx_in_batch[local_i]
                            if video_ids is not None and batch_i < len(video_ids):
                                cache_key = str(video_ids[batch_i])
                            if cache_enabled and cache_key is not None:
                                cache_queries += 1
                                cached = cache.get(cache_key, None)
                                if cached is not None:
                                    cache_hits += 1
                                    cache.move_to_end(cache_key)
                                    online_pooled_per_video[local_i] = cached.to(concat_images.device, non_blocking=True)
                                    continue
                            miss_images.append(vid_images)
                            miss_indices.append(local_i)
                            miss_split_sizes.append(vid_images.shape[0])

                        t1 = time.perf_counter()
                        if len(miss_images) > 0:
                            concat_miss_images = torch.cat(miss_images, dim=0).to(concat_images.device)
                            with torch.inference_mode():
                                miss_online_pooled = runtime_tower(concat_miss_images, split_sizes=miss_split_sizes)  # [Nmiss, Cg, 14, 14]
                            miss_online_splits = list(torch.split(miss_online_pooled, miss_split_sizes))
                            for miss_j, local_i in enumerate(miss_indices):
                                pooled = miss_online_splits[miss_j]
                                online_pooled_per_video[local_i] = pooled
                                batch_i = video_idx_in_batch[local_i]
                                if (
                                    cache_enabled
                                    and video_ids is not None
                                    and batch_i < len(video_ids)
                                ):
                                    cache_key = str(video_ids[batch_i])
                                    cache[cache_key] = pooled.detach().to("cpu", dtype=torch.bfloat16)
                                    while max_cache_items > 0 and len(cache) > max_cache_items:
                                        cache.popitem(last=False)
                        t2 = time.perf_counter()

                        projector = self.get_model().generative_projector
                        proj_param = next(projector.parameters())
                        proj_dtype = proj_param.dtype
                        projector_device = concat_images.device
                        need_backward_safe_input = torch.is_grad_enabled() and any(p.requires_grad for p in projector.parameters())
                        for local_i, batch_i in enumerate(video_idx_in_batch):
                            pooled = online_pooled_per_video[local_i]
                            Nv, Cg, Hg, Wg = pooled.shape
                            online_features = pooled.view(Nv, Cg, Hg * Wg).permute(0, 2, 1).contiguous()
                            online_features = online_features.to(device=projector_device, dtype=proj_dtype)
                            if need_backward_safe_input:
                                # Online towers run under inference mode; clone to normal tensors before trainable projector.
                                online_features = online_features.clone()
                            online_features = projector(online_features)
                            generative_features_list[batch_i] = online_features
                        t3 = time.perf_counter()
                        self._profile_generative(
                            tower=(t2 - t1),
                            projector=(t3 - t2),
                            cache_hits=cache_hits,
                            cache_queries=cache_queries,
                        )
                else:
                    raise ValueError(f"Unknown generative_feature_source: {generative_feature_source}")

            # image_features,all_faster_video_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            # This is a list, each element is [num_images, patch * patch, dim]
            # rank_print(f"Concat images : {concat_images.shape}")
            image_features = []
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    image_features.append(self.get_2dPool(image_feat))#[32, 196, 3584]
                else:
                    image_features.append(image_feat)
            # image_features:[32, 196, 3584]
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat") # spatial_unpad
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square") # anyres_max_9
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token") # grid

            new_generative_image_features = []
            if use_generative_feature and generative_features_list is not None:
                if use_feat_fusion:
                    for vid_idx in video_idx_in_batch:
                        if vid_idx < len(image_features) and vid_idx < len(generative_features_list) and generative_features_list[vid_idx] is not None:
                            instruction_ctx = None
                            if instruction_contexts is not None and vid_idx < len(instruction_contexts):
                                instruction_ctx = instruction_contexts[vid_idx]
                                if instruction_ctx is not None:
                                    instruction_ctx = instruction_ctx.to(
                                        device=image_features[vid_idx].device,
                                        dtype=image_features[vid_idx].dtype,
                                    )
                            image_features[vid_idx] = self.get_model().feature_fusion(
                                image_features[vid_idx],
                                generative_features_list[vid_idx],
                                instruction_ctx=instruction_ctx,
                            )
                else:
                    new_generative_image_features = []
                    for i, gen_feat in enumerate(generative_features_list):
                        if gen_feat is None:
                            new_generative_image_features.append(gen_feat)
                            continue
                        if "gen" in self.config.world_position_embedding_type:
                            coords = world_coords_discrete[0].flatten(1, 2)
                            gen_feat = gen_feat + self.get_model().world_position_embedding(coords.detach())
                        gen_feat = self.add_token_per_grid(gen_feat)
                        new_generative_image_features.append(gen_feat)
                    generative_features_list = new_generative_image_features

            if use_object_proposals:
                object_features = []
                valid_obj_num = 0
                for l in range(obj_num):
                    # print(f"image_featurs: {image_features[0].shape}")
                    # print(f"object_patch: {object_patch[l].shape}")
                    if "patch27" in self.config.object_feature_type:
                        cur_object_features = image_features[0][object_patch[l].view(-1, 196)]
                    elif "patch14" in self.config.object_feature_type:
                        cur_object_features = encoded_image_features[0][object_patch[l].view(-1, 729)]
                    else:
                        raise NotImplementedError

                    if len(cur_object_features) == 0:
                        cur_object_features = torch.zeros(image_features[0].shape[-1]).to(image_features[0].device)
                    else:
                        cur_object_features = cur_object_features.mean(dim=0)
                        valid_obj_num += 1
                    object_features.append(cur_object_features)
                object_features = torch.stack(object_features)
                if use_mlp_pe or use_sin3d_pe:
                    box_center_features = self.get_model().world_position_embedding(object_boxes_center.unsqueeze(0)).squeeze(0)      
                    object_features += box_center_features
            else:
                object_features =  None
            if use_sin3d_pe or use_mlp_pe:
                new_image_features = []
                for idx, image_feat in enumerate(image_features):
                    if "discrete" in self.config.world_position_embedding_type:
                        coords = world_coords_discrete[idx].flatten(1, 2)
                    else:
                        coords = world_coords[idx].flatten(1, 2)
                    image_feat = image_feat + self.get_model().world_position_embedding(coords.detach())
                    new_image_features.append(image_feat)
                image_features = new_image_features #[32, 196, 3584]

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    # import pdb; pdb.set_trace()
                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        if mm_newline_position == "grid":
                            # Grid-wise
                            image_feature = self.add_token_per_grid(image_feature) #[6720, 3584]
                            if getattr(self.config, "add_faster_video", False):
                                faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                                # Add a token for each frame
                                concat_slow_fater_token = []
                                # import pdb; pdb.set_trace()
                                for _ in range(image_feature.shape[0]):
                                    if _ % self.config.faster_token_stride == 0:
                                        concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                    else:
                                        concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                # import pdb; pdb.set_trace()
                                image_feature = torch.cat(concat_slow_fater_token)

                                # print("!!!!!!!!!!!!")
                        
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            image_feature = self.add_token_per_frame(image_feature)

                            new_image_features.append(image_feature.flatten(0, 1))
                            
                        elif mm_newline_position == "one_token":
                            # one-token
                            image_feature = image_feature.flatten(0, 1)
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                            new_image_features.append(image_feature)      
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        # rank0_print("Single-images")
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        new_image_features.append(image_feature)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                        new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")

            # Only concat generative features if gated fusion is not used
            if use_generative_feature and (use_feat_fusion != True) and generative_features_list is not None:
                new_image_features_with_gen = []
                for idx, img_feat in enumerate(image_features):
                    if idx in video_idx_in_batch and generative_features_list[idx] is not None:
                        gen_feat = generative_features_list[idx]  # [T, Lg, C]
                        if img_feat.dim() == 3:
                            img_feat = torch.cat([img_feat, gen_feat], dim=1)
                        else:
                            if gen_feat.dim() == 3:
                                gen_feat = gen_feat.flatten(0, 1)
                            img_feat = torch.cat([img_feat, gen_feat], dim=0)
                    new_image_features_with_gen.append(img_feat)
                image_features = new_image_features_with_gen
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        new_world_coords = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]

            cat_cur_input_ids_noim = torch.cat(cur_input_ids_noim)
            cur_input_embeds = self.get_model().embed_tokens(cat_cur_input_ids_noim)

            # Add input coord PE
            if hasattr(self.config, "coord_token_ids") and (use_sin3d_pe or use_mlp_pe):
                query_coord_tokens = (cat_cur_input_ids_noim == self.config.coord_token_ids[0])
                if query_coord_tokens.sum() != 0:
                    cur_input_embeds[query_coord_tokens] += self.get_model().world_position_embedding(box_input.unsqueeze(0).detach())[:, 0]

            
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_world_coords = []
            cur_pos_index = 0
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if use_mrope_position_embedding:
                    cur_new_world_coords.append(
                        torch.arange(cur_pos_index, cur_pos_index + len(cur_input_embeds_no_im[i])).to(cur_input_embeds_no_im[i].device).unsqueeze(1).repeat(1, 3)
                    )
                    cur_pos_index += len(cur_input_embeds_no_im[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    
                    if use_mrope_position_embedding:
                        coords = world_coords_discrete[batch_idx]
                        V, H, W, D = coords.shape
                        new_coords = torch.zeros(V*H*(W+1), 3).to(cur_input_embeds_no_im[i].device).view(V, H, W+1, 3)
                        new_coords[:, :, :W, :] = coords
                        new_coords = new_coords.view(-1, 3)
                        cur_pos_index += V * H * (W + 1)
                        cur_new_world_coords.append(new_coords)


                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

            if use_mrope_position_embedding:
                cur_new_world_coords = torch.cat(cur_new_world_coords, dim=0)
                new_world_coords.append(cur_new_world_coords)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        mrope_position_ids = torch.zeros((batch_size, max_len, 3), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    if use_mrope_position_embedding:
                        mrope_position_ids[i, -cur_len:, :] = new_world_coords[i][-cur_len:, :]

            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    if use_mrope_position_embedding:
                        mrope_position_ids[i, :cur_len, :] = new_world_coords[i][:cur_len, :]

        # mrope_position_ids = mrope_position_ids.permute(2, 0, 1)
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        
        if use_mrope_position_embedding:
            position_ids = mrope_position_ids

        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, object_features, object_boxes

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
