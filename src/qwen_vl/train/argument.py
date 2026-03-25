import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

    # Geometry encoder configuration
    use_geometry_encoder: bool = field(default=False)  # Whether to use 3D geometry encoder
    geometry_encoder_type: str = field(default="vggt")  # Type of geometry encoder ("vggt", "pi3")
    geometry_encoder_path: str = field(default="facebook/VGGT-1B/")  # Path to pre-trained geometry encoder model
    reference_frame: str = field(default="first")  # Reference frame for geometry encoding ("first", "last"), only available for vggt
    feature_fusion_method: str = field(default="add")  # Method to fuse geometry and visual features ("add", "concat", "cross_attention", "gated")
    fusion_num_layers: int = field(default=1)  # Number of layers in the cross-attention module when feature_fusion_method is "cross_attention"
    geometry_merger_type: str = field(default="mlp")  # Type of geometry feature merger ("mlp", "avg")

    # Generative encoder configuration
    use_generative_encoder: bool = field(default=False)  # Whether to use generative vision encoder
    generative_encoder_tower_type: str = field(default="wan_vace_online")  # Tower type (see multimodal_generative_encoder/builder.py)
    generative_encoder_path: Optional[str] = field(default=None)  # Checkpoint path for the generative tower
    generative_vision_tower_task: Optional[str] = field(default=None)  # WAN task override, e.g. "vace-1.3B" or "t2v-1.3B"
    generative_vision_tower_model_name: Optional[str] = field(default=None)  # Optional model name for online towers; tower-specific default is used when unset
    generative_encoder_feature_dim: int = field(default=2048)  # Feature dimension of the generative tower output
    generative_encoder_freeze: bool = field(default=True)  # Freeze generative encoder weights
    generative_fusion_method: str = field(default="token_gated_residual")  # Fusion method; the open-source release default is "token_gated_residual" (also supports "token_gated", "add", "gated", "concat", "cross_attention", "weighted")
    generative_fusion_num_layers: int = field(default=1)  # Number of fusion layers
    generative_merger_type: str = field(default="mlp")  # Type of merger ("mlp", "avg")

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    max_samples: int = field(default=-1)
    shuffle: bool = field(default=True)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
