import os
from typing import Dict, List, Tuple, Union, Any

import safetensors.torch
import torch
from huggingface_hub import hf_hub_download

from .model import Seva, SevaParams


def seed_everything(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _unwrap_checkpoint(obj: Any) -> Dict[str, torch.Tensor]:
    """
    兼容常见 checkpoint 包装格式：
      - {"state_dict": {...}}
      - {"model": {...}}
      - {"model_state_dict": {...}}
      - {"net": {...}}
      - {"module": {...}}
      - 直接就是 state_dict
    """
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "model_state_dict", "net", "module"):
            v = obj.get(k, None)
            if isinstance(v, dict) and any(isinstance(x, torch.Tensor) for x in v.values()):
                return v  # type: ignore[return-value]
        if any(isinstance(x, torch.Tensor) for x in obj.values()):
            return obj  # type: ignore[return-value]
    raise TypeError(f"Checkpoint format not understood: got {type(obj)}")

def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefixes=("module.", "model.", "net.")) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


def _filter_by_key_and_shape(
    src: Dict[str, torch.Tensor],
    dst_shapes: Dict[str, torch.Size],
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    kept = {}
    dropped = []
    for k, v in src.items():
        if k in dst_shapes and hasattr(v, "shape") and tuple(v.shape) == tuple(dst_shapes[k]):
            kept[k] = v
        else:
            dropped.append(k)
    return kept, dropped


def load_model_compatible(
    pretrained_model_name_or_path: str,
    weight_name: str,
    model_version: float = 1.1,
    device: Union[str, torch.device] = "cpu",
    verbose: bool = True,
) -> Seva:
    if os.path.isdir(pretrained_model_name_or_path):
        weight_path = os.path.join(pretrained_model_name_or_path, weight_name)
    else:
        final_name = weight_name
        if model_version > 1 and final_name.endswith(".safetensors"):
            base, ext = os.path.splitext(final_name)
            final_name = f"{base}v{model_version}{ext}"

        weight_path = hf_hub_download(repo_id=pretrained_model_name_or_path, filename=final_name)

        try:
            _ = hf_hub_download(repo_id=pretrained_model_name_or_path, filename="config.yaml")
        except Exception:
            pass

    if weight_path.endswith(".safetensors"):
        raw = safetensors.torch.load_file(weight_path, device=str(device))
    else:
        try:
            raw = torch.load(weight_path, map_location=str(device), weights_only=True)
        except TypeError:
            raw = torch.load(weight_path, map_location=str(device))

    state_dict = _unwrap_checkpoint(raw)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint format not understood: got {type(state_dict)}")

    state_dict = _strip_prefix(state_dict)

    with torch.device("meta"):
        model = Seva(SevaParams())

    dst_shapes = {k: v.shape for k, v in model.state_dict().items()}
    filtered, dropped = _filter_by_key_and_shape(state_dict, dst_shapes)

    missing, unexpected = model.load_state_dict(filtered, strict=False, assign=True)

    has_mismatch = (len(dropped) > 0) or (len(missing) > 0) or (len(unexpected) > 0)
    if verbose and has_mismatch:
        print(f"[load_model_compatible] weight_path: {weight_path}")
        print(f"[load_model_compatible] loaded tensors: {len(filtered)}")
        print(f"[load_model_compatible] dropped (key/shape mismatch): {len(dropped)}")
        print(f"[load_model_compatible] missing: {len(missing)}, unexpected(after filter): {len(unexpected)}")
        if len(missing) and len(missing) <= 30:
            print("  missing examples:", missing[:30])
        if len(dropped) and len(dropped) <= 30:
            print("  dropped examples:", dropped[:30])

    model = model.to(torch.bfloat16)
    return model
