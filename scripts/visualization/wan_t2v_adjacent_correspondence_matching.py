#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from llava.model.multimodal_generative_encoder.wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES
from llava.model.multimodal_generative_encoder.wan_t2v_encoder import WanT2VOnlineEncoder
from llava.video_utils import VideoProcessor

try:
    import matplotlib.cm as cm
except Exception:
    cm = None


def _normalize_scene_id(scene_id: str) -> str:
    s = str(scene_id).strip()
    if s.startswith("scannet/"):
        return s
    low = s.lower()
    if low.startswith("scene"):
        return f"scannet/{low}"
    if low.startswith("scannet_scene"):
        return f"scannet/{low.split('scannet_')[-1]}"
    return f"scannet/{low}"


def _load_scene_meta(annotation_dir: str, split: str, scene_id: str) -> Dict:
    pkl_path = os.path.join(annotation_dir, f"embodiedscan_infos_{split}.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Split metadata not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    for item in payload["data_list"]:
        if item.get("sample_idx") == scene_id:
            return item
    raise KeyError(f"Scene `{scene_id}` not found in split `{split}`.")


def _sample_frame_indices(total: int, num_frames: int) -> np.ndarray:
    if total <= 0:
        raise ValueError("No frames in scene metadata.")
    return np.linspace(0, total - 1, num_frames, dtype=int)


def _compute_resize_crop_params(orig_w: int, orig_h: int, out_w: int, out_h: int) -> Dict[str, float]:
    scale = max(out_h / float(orig_h), out_w / float(orig_w))
    new_h = max(1, int(round(orig_h * scale)))
    new_w = max(1, int(round(orig_w * scale)))
    top = max(0, (new_h - out_h) // 2)
    left = max(0, (new_w - out_w) // 2)
    sx = float(new_w) / float(orig_w)
    sy = float(new_h) / float(orig_h)
    return {
        "orig_w": float(orig_w),
        "orig_h": float(orig_h),
        "out_w": float(out_w),
        "out_h": float(out_h),
        "sx": float(sx),
        "sy": float(sy),
        "new_w": float(new_w),
        "new_h": float(new_h),
        "left": float(left),
        "top": float(top),
    }


def _token_center_original(row: int, col: int, params: Dict[str, float], grid_size: int = 14) -> Tuple[float, float]:
    out_w = float(params["out_w"])
    out_h = float(params["out_h"])
    left = float(params["left"])
    top = float(params["top"])
    sx = float(params["sx"])
    sy = float(params["sy"])
    orig_w = float(params["orig_w"])
    orig_h = float(params["orig_h"])

    cxp = (float(col) + 0.5) * (out_w / float(grid_size))
    cyp = (float(row) + 0.5) * (out_h / float(grid_size))
    cxo = (cxp + left) / sx
    cyo = (cyp + top) / sy
    cxo = max(0.0, min(orig_w - 1.0, cxo))
    cyo = max(0.0, min(orig_h - 1.0, cyo))
    return float(cxo), float(cyo)


def _jet_heatmap(score_map: np.ndarray) -> np.ndarray:
    score = np.asarray(score_map, dtype=np.float32)
    score = score - score.min()
    score = score / (score.max() + 1e-8)
    if cm is not None:
        heat = cm.get_cmap("turbo")(score)[..., :3]
    else:
        r = np.clip(1.5 - np.abs(4.0 * score - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * score - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * score - 1.0), 0.0, 1.0)
        heat = np.stack([r, g, b], axis=-1)
    return (heat * 255.0 + 0.5).astype(np.uint8)


def _overlay(base_rgb: np.ndarray, heat_rgb: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    base = base_rgb.astype(np.float32)
    heat = heat_rgb.astype(np.float32)
    out = (1.0 - alpha) * base + float(alpha) * heat
    return np.clip(out, 0, 255).astype(np.uint8)


def _upsample_14_to_hw(grid14: np.ndarray, h: int, w: int) -> np.ndarray:
    t = torch.from_numpy(np.asarray(grid14, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    y = F.interpolate(t, size=(int(h), int(w)), mode="bilinear", align_corners=False)
    return y[0, 0].numpy()


def _encode_wan_t2v_prepool_features(
    frames_unit: torch.Tensor,
    checkpoint_dir: str,
    task: str,
    size: str,
    timestep: int,
    shift: float,
    feat_block_idx: int,
    dtype_name: str,
    device: torch.device,
    chunk_size: int,
    seed: int,
) -> torch.Tensor:
    cfg = SimpleNamespace(
        generative_vision_tower_task=task,
        generative_vision_tower_checkpoint=checkpoint_dir,
        generative_vision_tower_size=size,
        generative_vision_tower_timestep=timestep,
        generative_vision_tower_shift=shift,
        generative_vision_tower_feat_block_idx=feat_block_idx,
        generative_vision_tower_dtype=dtype_name,
    )
    encoder = WanT2VOnlineEncoder(cfg).eval()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    encoder._move_models_to_device(device)
    block_idx = (len(encoder.model.blocks) - 1) if encoder.feat_block_idx < 0 else encoder.feat_block_idx
    if block_idx < 0 or block_idx >= len(encoder.model.blocks):
        raise ValueError(f"feat_block_idx out of range: {block_idx}")

    outs = []
    n = int(frames_unit.shape[0])
    step = max(1, int(chunk_size))
    use_autocast = encoder.param_dtype in (torch.float16, torch.bfloat16)
    with torch.inference_mode():
        for st in range(0, n, step):
            ed = min(n, st + step)
            x = frames_unit[st:ed].to(device=device, dtype=torch.float32, non_blocking=True)
            if x.shape[0] == 0:
                continue

            x_proc = encoder._prepare_frames(x)
            frame_list = [x_proc[i].unsqueeze(1) for i in range(x_proc.shape[0])]
            encoder._ensure_scheduler_ready(device)
            tau = encoder._select_timestep(encoder.scheduler.timesteps, target_timestep=int(encoder.timestep))
            context = encoder._get_text_context(device=device, batch_size=len(frame_list))

            with torch.autocast(device_type=device.type, dtype=encoder.param_dtype, enabled=use_autocast):
                base_latents = encoder.vae.encode(frame_list)
                target_shape = list(base_latents[0].shape)
                seq_len = math.ceil(
                    (target_shape[2] * target_shape[3]) / (encoder.patch_size[1] * encoder.patch_size[2]) * target_shape[1]
                )

                latent_batch = torch.stack(base_latents, dim=0)
                noise = torch.randn_like(latent_batch)
                noisy_latents = encoder.scheduler.add_noise(
                    original_samples=latent_batch,
                    noise=noise,
                    timesteps=tau.expand(len(frame_list)),
                )
                noisy_latents_list = [noisy_latents[i] for i in range(len(frame_list))]

                feat_holder: Dict[str, torch.Tensor] = {}

                def _hook(_, __, output):
                    feat_holder["feat"] = output.detach()

                handle = encoder.model.blocks[block_idx].register_forward_hook(_hook)
                try:
                    t = tau.expand(len(frame_list)).to(device=device, dtype=torch.long)
                    _ = encoder.model(
                        noisy_latents_list,
                        t=t,
                        context=context,
                        seq_len=seq_len,
                    )
                finally:
                    handle.remove()

                if "feat" not in feat_holder:
                    raise RuntimeError("Failed to capture WAN-T2V pre-pooling features.")

                feats = feat_holder["feat"]  # [N, L, C]
                grid_h = encoder.frame_height // (encoder.vae_stride[1] * encoder.patch_size[1])
                grid_w = encoder.frame_width // (encoder.vae_stride[2] * encoder.patch_size[2])
                tokens_per_frame = grid_h * grid_w
                if feats.shape[1] != tokens_per_frame:
                    raise RuntimeError(
                        f"Unexpected token count: {feats.shape[1]} (expected {tokens_per_frame})."
                    )
                feats = feats.view(feats.shape[0], grid_h, grid_w, feats.shape[2]).permute(0, 3, 1, 2).contiguous()

            outs.append(feats.detach().cpu())
            del x, x_proc, frame_list, base_latents, latent_batch, noise, noisy_latents, noisy_latents_list, feats
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if len(outs) == 0:
        return torch.empty((0, int(encoder.cfg.dim), 0, 0), dtype=torch.float32)
    return torch.cat(outs, dim=0)


def _resize_center_crop_coords(
    world_coords: torch.Tensor,  # [T,H,W,3]
    out_h: int,
    out_w: int,
) -> torch.Tensor:
    x = world_coords.permute(0, 3, 1, 2).float()
    _, _, h, w = x.shape
    scale = max(out_h / float(h), out_w / float(w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    x = F.interpolate(x, size=(new_h, new_w), mode="nearest")
    top = max(0, (new_h - out_h) // 2)
    left = max(0, (new_w - out_w) // 2)
    x = x[:, :, top : top + out_h, left : left + out_w]
    return x.permute(0, 2, 3, 1).contiguous()


def _resize_center_crop_mask(mask: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    x = mask.float().unsqueeze(1)
    _, _, h, w = x.shape
    scale = max(out_h / float(h), out_w / float(w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    x = F.interpolate(x, size=(new_h, new_w), mode="nearest")
    top = max(0, (new_h - out_h) // 2)
    left = max(0, (new_w - out_w) // 2)
    x = x[:, :, top : top + out_h, left : left + out_w]
    return x[:, 0] > 0.5


def _load_depth_valid_masks(frame_files: List[str]) -> torch.Tensor:
    masks = []
    for p in frame_files:
        dp = p.replace(".jpg", ".png")
        if not os.path.exists(dp):
            dp = os.path.splitext(p)[0] + ".png"
        if not os.path.exists(dp):
            raise FileNotFoundError(f"Depth file not found: {dp}")
        with Image.open(dp) as dimg:
            d = np.array(dimg, copy=True)
        masks.append(torch.from_numpy((d > 0).astype(np.uint8)))
    return torch.stack(masks, dim=0).bool()


def _pair_voxel_correspondence(
    feat_i: torch.Tensor,  # [14,14,C]
    feat_j: torch.Tensor,  # [14,14,C]
    xyz_i: torch.Tensor,  # [14,14,3]
    xyz_j: torch.Tensor,  # [14,14,3]
    valid_i: torch.Tensor,  # [14,14]
    valid_j: torch.Tensor,  # [14,14]
    voxel_size: float,
) -> Dict:
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be > 0, got {voxel_size}")

    fi = feat_i.reshape(-1, feat_i.shape[-1]).float()
    fj = feat_j.reshape(-1, feat_j.shape[-1]).float()
    xi = xyz_i.reshape(-1, 3).float()
    xj = xyz_j.reshape(-1, 3).float()
    vi = valid_i.reshape(-1).bool()
    vj = valid_j.reshape(-1).bool()

    fi = F.normalize(fi, dim=-1)
    fj = F.normalize(fj, dim=-1)

    all_xyz = torch.cat([xi[vi], xj[vj]], dim=0)
    if all_xyz.shape[0] == 0:
        raise RuntimeError("No valid xyz for pair correspondence.")
    xyz_min = all_xyz.amin(dim=0, keepdim=True)

    voxel_i = torch.floor((xi - xyz_min) / float(voxel_size)).to(torch.int64)
    voxel_j = torch.floor((xj - xyz_min) / float(voxel_size)).to(torch.int64)

    # key -> {"i":[token_ids], "j":[token_ids]}
    buckets: Dict[Tuple[int, int, int], Dict[str, List[int]]] = {}
    for idx in torch.nonzero(vi, as_tuple=False).view(-1).tolist():
        key = (int(voxel_i[idx, 0].item()), int(voxel_i[idx, 1].item()), int(voxel_i[idx, 2].item()))
        if key not in buckets:
            buckets[key] = {"i": [], "j": []}
        buckets[key]["i"].append(int(idx))
    for idx in torch.nonzero(vj, as_tuple=False).view(-1).tolist():
        key = (int(voxel_j[idx, 0].item()), int(voxel_j[idx, 1].item()), int(voxel_j[idx, 2].item()))
        if key not in buckets:
            buckets[key] = {"i": [], "j": []}
        buckets[key]["j"].append(int(idx))

    map_i = torch.full((14, 14), float("nan"), dtype=torch.float32)
    map_j = torch.full((14, 14), float("nan"), dtype=torch.float32)
    pair_sims = []
    num_voxels_with_pairs = 0
    for key, entry in buckets.items():
        _ = key
        if len(entry["i"]) == 0 or len(entry["j"]) == 0:
            continue
        num_voxels_with_pairs += 1
        pi = F.normalize(fi[entry["i"]].mean(dim=0), dim=0)
        pj = F.normalize(fj[entry["j"]].mean(dim=0), dim=0)
        sim = float(torch.dot(pi, pj).item())
        pair_sims.append(sim)
        for tid in entry["i"]:
            r, c = divmod(int(tid), 14)
            map_i[r, c] = sim
        for tid in entry["j"]:
            r, c = divmod(int(tid), 14)
            map_j[r, c] = sim

    score = float(sum(pair_sims) / len(pair_sims)) if len(pair_sims) > 0 else float("nan")
    return {
        "score": score,
        "num_pairs": int(len(pair_sims)),
        "num_voxels_with_pairs": int(num_voxels_with_pairs),
        "num_voxels_total": int(len(buckets)),
        "map_i": map_i,
        "map_j": map_j,
    }


def _pair_feature_matching(
    feat_i: torch.Tensor,  # [14,14,C]
    feat_j: torch.Tensor,  # [14,14,C]
    xyz_i: torch.Tensor,  # [14,14,3]
    xyz_j: torch.Tensor,  # [14,14,3]
    valid_i: torch.Tensor,  # [14,14]
    valid_j: torch.Tensor,  # [14,14]
    sim_threshold: float,
    mutual_only: bool,
    max_world_dist: float,
) -> Dict:
    fi = feat_i.reshape(-1, feat_i.shape[-1]).float()
    fj = feat_j.reshape(-1, feat_j.shape[-1]).float()
    xi = xyz_i.reshape(-1, 3).float()
    xj = xyz_j.reshape(-1, 3).float()
    vi = valid_i.reshape(-1).bool()
    vj = valid_j.reshape(-1).bool()

    idx_i = torch.nonzero(vi, as_tuple=False).view(-1)
    idx_j = torch.nonzero(vj, as_tuple=False).view(-1)
    if idx_i.numel() == 0 or idx_j.numel() == 0:
        return {
            "matches": [],
            "num_valid_i": int(idx_i.numel()),
            "num_valid_j": int(idx_j.numel()),
            "num_mutual": 0,
            "num_selected": 0,
            "mean_sim": float("nan"),
            "mean_world_dist": float("nan"),
        }

    fi_v = F.normalize(fi[idx_i], dim=-1)  # [Ni,C]
    fj_v = F.normalize(fj[idx_j], dim=-1)  # [Nj,C]
    sim = fi_v @ fj_v.t()  # [Ni,Nj]
    s_i, nn_j = sim.max(dim=1)  # [Ni]
    s_j, nn_i = sim.max(dim=0)  # [Nj]

    matches = []
    num_mutual = 0
    for i_local in range(idx_i.numel()):
        j_local = int(nn_j[i_local].item())
        cur_sim = float(s_i[i_local].item())
        is_mutual = int(nn_i[j_local].item()) == int(i_local)
        if is_mutual:
            num_mutual += 1
        if mutual_only and not is_mutual:
            continue
        if cur_sim < float(sim_threshold):
            continue

        ti = int(idx_i[i_local].item())
        tj = int(idx_j[j_local].item())
        wi = xi[ti]
        wj = xj[tj]
        wdist = float(torch.norm(wi - wj, p=2).item())
        if float(max_world_dist) > 0 and wdist > float(max_world_dist):
            continue

        ri, ci = divmod(ti, 14)
        rj, cj = divmod(tj, 14)
        matches.append(
            {
                "ti": int(ti),
                "tj": int(tj),
                "ri": int(ri),
                "ci": int(ci),
                "rj": int(rj),
                "cj": int(cj),
                "sim": float(cur_sim),
                "world_dist": float(wdist),
                "mutual": bool(is_mutual),
            }
        )

    matches.sort(key=lambda x: x["sim"], reverse=True)
    sims = [m["sim"] for m in matches]
    dists = [m["world_dist"] for m in matches]
    mean_sim = float(sum(sims) / len(sims)) if len(sims) > 0 else float("nan")
    mean_dist = float(sum(dists) / len(dists)) if len(dists) > 0 else float("nan")
    return {
        "matches": matches,
        "num_valid_i": int(idx_i.numel()),
        "num_valid_j": int(idx_j.numel()),
        "num_mutual": int(num_mutual),
        "num_selected": int(len(matches)),
        "mean_sim": mean_sim,
        "mean_world_dist": mean_dist,
    }


def _draw_match_panel(
    img_i: Image.Image,
    img_j: Image.Image,
    params_i: Dict[str, float],
    params_j: Dict[str, float],
    matches: List[Dict],
    num_draw: int,
    line_width: int,
    point_radius: int,
) -> np.ndarray:
    left = np.asarray(img_i, dtype=np.uint8)
    right = np.asarray(img_j, dtype=np.uint8)
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1]
    panel = Image.new("RGB", (w, h), color=(255, 255, 255))
    panel.paste(Image.fromarray(left), (0, 0))
    panel.paste(Image.fromarray(right), (left.shape[1], 0))
    draw = ImageDraw.Draw(panel)

    k = min(int(num_draw), len(matches))
    for m in matches[:k]:
        x1, y1 = _token_center_original(m["ri"], m["ci"], params_i, grid_size=14)
        x2, y2 = _token_center_original(m["rj"], m["cj"], params_j, grid_size=14)
        x2 += float(left.shape[1])

        s = float(m["sim"])
        s01 = max(0.0, min(1.0, (s + 1.0) * 0.5))
        if cm is not None:
            col = cm.get_cmap("viridis")(s01)[:3]
            color = tuple(int(round(v * 255.0)) for v in col)
        else:
            color = (int(255 * (1 - s01)), int(255 * s01), 150)

        draw.line([x1, y1, x2, y2], fill=color, width=int(line_width))
        r = int(point_radius)
        draw.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=color)
        draw.ellipse([x2 - r, y2 - r, x2 + r, y2 + r], fill=color)
    return np.asarray(panel, dtype=np.uint8)


def _write_csv(path: Path, rows: List[Dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _compute_feature_rgb_maps(feat_tokens: torch.Tensor) -> Tuple[np.ndarray, Dict]:
    # feat_tokens: [T,H,W,C] -> rgb: [T,H,W,3], uint8
    t, h, w, c = feat_tokens.shape
    x = feat_tokens.reshape(-1, c).float()  # [N,C]

    info = {"method": None}
    try:
        # PCA projection gives semantically smoother feature visualization than channel picking.
        _, _, v = torch.pca_lowrank(x, q=3, center=True, niter=4)
        centered = x - x.mean(dim=0, keepdim=True)
        proj = centered @ v[:, :3]  # [N,3]
        info["method"] = "pca"
    except Exception:
        var = x.var(dim=0)
        top3 = torch.topk(var, k=3).indices
        proj = x[:, top3]
        info["method"] = "topvar_fallback"
        info["top_channels"] = [int(v.item()) for v in top3]

    q_low = torch.quantile(proj, q=0.01, dim=0)
    q_high = torch.quantile(proj, q=0.99, dim=0)
    proj_n = (proj - q_low[None, :]) / (q_high[None, :] - q_low[None, :] + 1e-6)
    proj_n = proj_n.clamp(0, 1)
    rgb = (proj_n * 255.0 + 0.5).to(torch.uint8).view(t, h, w, 3).cpu().numpy()
    info["q_low"] = [float(v.item()) for v in q_low]
    info["q_high"] = [float(v.item()) for v in q_high]
    return rgb, info


def _compute_feature_norm_vis(feat_tokens: torch.Tensor, pmin: float = 1.0, pmax: float = 99.0) -> np.ndarray:
    # feat_tokens: [T,H,W,C] -> [T,H,W] in [0,1]
    norm = torch.norm(feat_tokens.float(), dim=-1).cpu().numpy()
    vals = norm[np.isfinite(norm)]
    if vals.size == 0:
        return np.zeros_like(norm, dtype=np.float32)
    lo = float(np.percentile(vals, pmin))
    hi = float(np.percentile(vals, pmax))
    if hi <= lo:
        hi = lo + 1e-6
    out = np.clip((norm - lo) / (hi - lo), 0.0, 1.0)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def _draw_grid_lines(
    img: Image.Image,
    grid_h: int = 14,
    grid_w: int = 14,
    color: Tuple[int, int, int] = (45, 45, 45),
) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for k in range(1, int(grid_w)):
        x = int(round(k * w / float(grid_w)))
        draw.line([(x, 0), (x, h)], fill=color, width=1)
    for k in range(1, int(grid_h)):
        y = int(round(k * h / float(grid_h)))
        draw.line([(0, y), (w, y)], fill=color, width=1)
    return out


def _token_center_grid(row: int, col: int, width: int, height: int, grid_h: int = 14, grid_w: int = 14) -> Tuple[float, float]:
    x = (float(col) + 0.5) * (float(width) / float(grid_w))
    y = (float(row) + 0.5) * (float(height) / float(grid_h))
    return float(x), float(y)


def _draw_feature_match_panel(
    feat_rgb_i: np.ndarray,
    feat_rgb_j: np.ndarray,
    matches: List[Dict],
    num_draw: int,
    line_width: int,
    point_radius: int,
    draw_grid: bool,
    vis_grid_h: int,
    vis_grid_w: int,
    match_grid_h: int = 14,
    match_grid_w: int = 14,
) -> np.ndarray:
    left_img = Image.fromarray(feat_rgb_i)
    right_img = Image.fromarray(feat_rgb_j)
    if draw_grid:
        left_img = _draw_grid_lines(left_img, grid_h=vis_grid_h, grid_w=vis_grid_w, color=(60, 60, 60))
        right_img = _draw_grid_lines(right_img, grid_h=vis_grid_h, grid_w=vis_grid_w, color=(60, 60, 60))

    left = np.asarray(left_img, dtype=np.uint8)
    right = np.asarray(right_img, dtype=np.uint8)
    gap = 24
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + gap + right.shape[1]
    panel = Image.new("RGB", (w, h), color=(240, 240, 240))
    panel.paste(Image.fromarray(left), (0, 0))
    panel.paste(Image.fromarray(right), (left.shape[1] + gap, 0))
    draw = ImageDraw.Draw(panel)

    k = min(int(num_draw), len(matches))
    for m in matches[:k]:
        x1, y1 = _token_center_grid(
            m["ri"],
            m["ci"],
            width=left.shape[1],
            height=left.shape[0],
            grid_h=match_grid_h,
            grid_w=match_grid_w,
        )
        x2, y2 = _token_center_grid(
            m["rj"],
            m["cj"],
            width=right.shape[1],
            height=right.shape[0],
            grid_h=match_grid_h,
            grid_w=match_grid_w,
        )
        x2 += float(left.shape[1] + gap)

        s = float(m["sim"])
        s01 = max(0.0, min(1.0, (s + 1.0) * 0.5))
        if cm is not None:
            col = cm.get_cmap("viridis")(s01)[:3]
            color = tuple(int(round(v * 255.0)) for v in col)
        else:
            color = (int(255 * (1 - s01)), int(255 * s01), 150)

        draw.line([x1, y1, x2, y2], fill=color, width=int(line_width))
        r = int(point_radius)
        draw.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=color)
        draw.ellipse([x2 - r, y2 - r, x2 + r, y2 + r], fill=color)
    return np.asarray(panel, dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(
        description="WAN adjacent-frame correspondence score + token feature matching visualization."
    )
    parser.add_argument("--scene_id", type=str, default="scannet/scene0000_00")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--annotation_dir", type=str, default="data/embodiedscan")
    parser.add_argument("--video_folder", type=str, default="data")
    parser.add_argument("--checkpoint_dir", type=str, default="data/models/Wan2.1-T2V-1.3B")
    parser.add_argument("--task", type=str, default="t2v-1.3B")
    parser.add_argument("--size", type=str, default="832*480")
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--feat_block_idx", type=int, default=20)
    parser.add_argument("--timestep", type=int, default=300)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--valid_ratio_threshold", type=float, default=0.1)
    parser.add_argument("--heatmap_alpha", type=float, default=0.55)
    parser.add_argument("--norm_pmin", type=float, default=5.0)
    parser.add_argument("--norm_pmax", type=float, default=95.0)
    parser.add_argument("--heatmap_gamma", type=float, default=1.2)
    parser.add_argument("--match_mutual_only", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--match_sim_threshold", type=float, default=0.60)
    parser.add_argument("--match_max_world_dist", type=float, default=0.0, help="<=0 means disabled.")
    parser.add_argument("--num_matches_draw", type=int, default=70)
    parser.add_argument("--line_width", type=int, default=2)
    parser.add_argument("--point_radius", type=int, default=3)
    parser.add_argument("--feature_map_scale", type=int, default=32, help="Each pre-pooling token is enlarged to this many pixels for feature-space visualization.")
    parser.add_argument("--feature_draw_grid", type=str, default="true", choices=["true", "false"])
    parser.add_argument("--output_dir", type=str, default="results/wan_t2v_adjacent_matching/scene0000_00")
    return parser.parse_args()


def run(args):
    scene_id = _normalize_scene_id(args.scene_id)

    if args.task not in SUPPORTED_SIZES:
        raise ValueError(f"Unsupported WAN task: {args.task}")
    if args.size not in SIZE_CONFIGS:
        raise ValueError(f"Unsupported size: {args.size}")
    if args.size not in SUPPORTED_SIZES[args.task]:
        raise ValueError(f"Size `{args.size}` not supported by task `{args.task}`: {SUPPORTED_SIZES[args.task]}")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("This script requires CUDA for Wan2.1-T2V inference.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    out_root = Path(args.output_dir)
    out_frames = out_root / "frames"
    out_corr = out_root / "adjacent_correspondence"
    out_match = out_root / "adjacent_matching"
    out_feat = out_root / "feature_maps"
    out_feat_match = out_root / "adjacent_feature_matching"
    out_tensors = out_root / "tensors"
    for p in [out_frames, out_corr, out_match, out_feat, out_feat_match, out_tensors]:
        p.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))

    scene_meta = _load_scene_meta(args.annotation_dir, args.split, scene_id)
    sample_idx = _sample_frame_indices(total=len(scene_meta["images"]), num_frames=int(args.num_frames))
    sampled_images = [scene_meta["images"][int(i)] for i in sample_idx]
    frame_files = [os.path.join(args.video_folder, x["img_path"]) for x in sampled_images]

    raw_images: List[Image.Image] = []
    resize_params: List[Dict[str, float]] = []
    frame_w, frame_h = SIZE_CONFIGS[args.size]
    for i, frame_path in enumerate(frame_files):
        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"Frame file not found: {frame_path}")
        with Image.open(frame_path) as im:
            rgb = im.convert("RGB")
        raw_images.append(rgb)
        resize_params.append(_compute_resize_crop_params(rgb.size[0], rgb.size[1], out_w=frame_w, out_h=frame_h))
        rgb.save(out_frames / f"frame_{i:03d}_raw.png")

    raw_np = np.stack([np.asarray(im, dtype=np.uint8) for im in raw_images], axis=0)
    frames_unit = torch.from_numpy(raw_np).permute(0, 3, 1, 2).float() / 255.0
    feats_prepool = _encode_wan_t2v_prepool_features(
        frames_unit=frames_unit,
        checkpoint_dir=args.checkpoint_dir,
        task=args.task,
        size=args.size,
        timestep=int(args.timestep),
        shift=float(args.shift),
        feat_block_idx=int(args.feat_block_idx),
        dtype_name=args.dtype,
        device=device,
        chunk_size=int(args.chunk_size),
        seed=int(args.seed),
    ).float()  # [T,C,Hpre,Wpre]
    if feats_prepool.ndim != 4 or feats_prepool.shape[0] != int(args.num_frames):
        raise RuntimeError(f"Unexpected WAN pre-pooling feature shape: {tuple(feats_prepool.shape)}")

    feats14 = F.adaptive_avg_pool2d(feats_prepool, output_size=(14, 14))
    if feats14.ndim != 4 or feats14.shape[2:] != (14, 14):
        raise RuntimeError(f"Unexpected WAN feature shape: {tuple(feats14.shape)}")

    vp = VideoProcessor(
        video_folder=args.video_folder,
        annotation_dir=args.annotation_dir,
        frame_sampling_strategy="uniform",
        val_box_type="pred",
    )
    world_raw = vp.calculate_world_coords(scene_id, frame_files)["world_coords"].float()  # [T,H,W,3]
    depth_valid_raw = _load_depth_valid_masks(frame_files)  # [T,H,W]

    world_proc = _resize_center_crop_coords(world_raw, out_h=frame_h, out_w=frame_w)
    valid_proc = _resize_center_crop_mask(depth_valid_raw, out_h=frame_h, out_w=frame_w)
    world14 = F.adaptive_avg_pool2d(world_proc.permute(0, 3, 1, 2), output_size=(14, 14)).permute(0, 2, 3, 1).contiguous()
    valid14_ratio = F.adaptive_avg_pool2d(valid_proc.float().unsqueeze(1), output_size=(14, 14))[:, 0]
    valid14 = valid14_ratio >= float(args.valid_ratio_threshold)
    feat_tokens = feats14.permute(0, 2, 3, 1).contiguous()  # [T,14,14,C] for correspondence/matching.
    feat_tokens_prepool = feats_prepool.permute(0, 2, 3, 1).contiguous()  # [T,Hpre,Wpre,C] for PCA/norm visualization.

    mutual_only = str(args.match_mutual_only).lower() == "true"
    feature_draw_grid = str(args.feature_draw_grid).lower() == "true"
    feature_map_scale = max(4, int(args.feature_map_scale))

    # Feature-space visualization maps (global PCA colorization + norm map) from pre-pooling resolution.
    feature_rgb_prepool, feature_rgb_info = _compute_feature_rgb_maps(feat_tokens_prepool)
    feature_norm_prepool = _compute_feature_norm_vis(feat_tokens_prepool, pmin=1.0, pmax=99.0)
    feature_grid_h = int(feat_tokens_prepool.shape[1])
    feature_grid_w = int(feat_tokens_prepool.shape[2])
    feature_map_h = feature_grid_h * feature_map_scale
    feature_map_w = feature_grid_w * feature_map_scale
    for i in range(int(args.num_frames)):
        feat_rgb = Image.fromarray(feature_rgb_prepool[i]).resize((feature_map_w, feature_map_h), resample=Image.NEAREST)
        if feature_draw_grid:
            feat_rgb = _draw_grid_lines(feat_rgb, grid_h=feature_grid_h, grid_w=feature_grid_w, color=(70, 70, 70))
        feat_rgb_np = np.asarray(feat_rgb, dtype=np.uint8)

        norm_pre = feature_norm_prepool[i]
        norm_rgb_pre = _jet_heatmap(norm_pre)
        norm_img = Image.fromarray(norm_rgb_pre).resize((feature_map_w, feature_map_h), resample=Image.NEAREST)
        if feature_draw_grid:
            norm_img = _draw_grid_lines(norm_img, grid_h=feature_grid_h, grid_w=feature_grid_w, color=(70, 70, 70))
        norm_np = np.asarray(norm_img, dtype=np.uint8)

        Image.fromarray(feat_rgb_np).save(out_feat / f"frame_{i:03d}_feature_rgb.png")
        Image.fromarray(norm_np).save(out_feat / f"frame_{i:03d}_feature_norm.png")
        Image.fromarray(np.concatenate([feat_rgb_np, norm_np], axis=1)).save(out_feat / f"frame_{i:03d}_feature_panel.png")

    pair_rows = []
    pair_corr_maps = []
    pair_match_data = []
    for i in range(int(args.num_frames) - 1):
        corr = _pair_voxel_correspondence(
            feat_i=feat_tokens[i],
            feat_j=feat_tokens[i + 1],
            xyz_i=world14[i],
            xyz_j=world14[i + 1],
            valid_i=valid14[i],
            valid_j=valid14[i + 1],
            voxel_size=float(args.voxel_size),
        )
        match = _pair_feature_matching(
            feat_i=feat_tokens[i],
            feat_j=feat_tokens[i + 1],
            xyz_i=world14[i],
            xyz_j=world14[i + 1],
            valid_i=valid14[i],
            valid_j=valid14[i + 1],
            sim_threshold=float(args.match_sim_threshold),
            mutual_only=bool(mutual_only),
            max_world_dist=float(args.match_max_world_dist),
        )

        pair_rows.append(
            {
                "pair_idx": int(i),
                "frame_i": int(i),
                "frame_j": int(i + 1),
                "sample_index_i": int(sample_idx[i]),
                "sample_index_j": int(sample_idx[i + 1]),
                "corr_score": float(corr["score"]),
                "corr_num_pairs": int(corr["num_pairs"]),
                "corr_num_voxels_with_pairs": int(corr["num_voxels_with_pairs"]),
                "corr_num_voxels_total": int(corr["num_voxels_total"]),
                "match_num_valid_i": int(match["num_valid_i"]),
                "match_num_valid_j": int(match["num_valid_j"]),
                "match_num_mutual": int(match["num_mutual"]),
                "match_num_selected": int(match["num_selected"]),
                "match_mean_sim": float(match["mean_sim"]),
                "match_mean_world_dist": float(match["mean_world_dist"]),
                "match_best_sim": float(match["matches"][0]["sim"]) if len(match["matches"]) > 0 else float("nan"),
            }
        )
        pair_corr_maps.append((corr["map_i"], corr["map_j"]))
        pair_match_data.append(match["matches"])

    _write_csv(
        out_root / "adjacent_pair_metrics.csv",
        pair_rows,
        fieldnames=[
            "pair_idx",
            "frame_i",
            "frame_j",
            "sample_index_i",
            "sample_index_j",
            "corr_score",
            "corr_num_pairs",
            "corr_num_voxels_with_pairs",
            "corr_num_voxels_total",
            "match_num_valid_i",
            "match_num_valid_j",
            "match_num_mutual",
            "match_num_selected",
            "match_mean_sim",
            "match_mean_world_dist",
            "match_best_sim",
        ],
    )

    # Global normalization for correspondence overlays.
    finite_vals = []
    for m0, m1 in pair_corr_maps:
        a = m0.numpy()
        b = m1.numpy()
        finite_vals.append(a[np.isfinite(a)])
        finite_vals.append(b[np.isfinite(b)])
    finite_concat = np.concatenate([x for x in finite_vals if x.size > 0], axis=0) if len(finite_vals) > 0 else np.array([], dtype=np.float32)
    if finite_concat.size == 0:
        lo, hi = 0.0, 1.0
    else:
        pmin = float(args.norm_pmin)
        pmax = float(args.norm_pmax)
        if pmax <= pmin:
            raise ValueError(f"norm_pmax must be > norm_pmin, got {pmax} <= {pmin}")
        lo = float(np.percentile(finite_concat, pmin))
        hi = float(np.percentile(finite_concat, pmax))
        if hi <= lo:
            hi = lo + 1e-6

    # Visualize each adjacent pair: correspondence overlay + matching panel.
    for i in range(int(args.num_frames) - 1):
        map_i = pair_corr_maps[i][0].numpy()
        map_j = pair_corr_maps[i][1].numpy()
        map_i = np.where(np.isfinite(map_i), map_i, lo)
        map_j = np.where(np.isfinite(map_j), map_j, lo)
        map_i = np.power(np.clip((map_i - lo) / (hi - lo), 0.0, 1.0), float(args.heatmap_gamma))
        map_j = np.power(np.clip((map_j - lo) / (hi - lo), 0.0, 1.0), float(args.heatmap_gamma))

        img_i = np.asarray(raw_images[i], dtype=np.uint8)
        img_j = np.asarray(raw_images[i + 1], dtype=np.uint8)
        heat_i = _jet_heatmap(_upsample_14_to_hw(map_i, img_i.shape[0], img_i.shape[1]))
        heat_j = _jet_heatmap(_upsample_14_to_hw(map_j, img_j.shape[0], img_j.shape[1]))
        ov_i = _overlay(img_i, heat_i, alpha=float(args.heatmap_alpha))
        ov_j = _overlay(img_j, heat_j, alpha=float(args.heatmap_alpha))
        corr_panel = np.concatenate([ov_i, ov_j], axis=1)
        Image.fromarray(corr_panel).save(out_corr / f"pair_{i:03d}_correspondence_overlay.png")

        match_panel = _draw_match_panel(
            img_i=raw_images[i],
            img_j=raw_images[i + 1],
            params_i=resize_params[i],
            params_j=resize_params[i + 1],
            matches=pair_match_data[i],
            num_draw=int(args.num_matches_draw),
            line_width=int(args.line_width),
            point_radius=int(args.point_radius),
        )
        Image.fromarray(match_panel).save(out_match / f"pair_{i:03d}_feature_matching.png")

        # Feature-space matching panel (lines on feature maps instead of original frames).
        feat_i = Image.fromarray(feature_rgb_prepool[i]).resize((feature_map_w, feature_map_h), resample=Image.NEAREST)
        feat_j = Image.fromarray(feature_rgb_prepool[i + 1]).resize((feature_map_w, feature_map_h), resample=Image.NEAREST)
        feat_match_panel = _draw_feature_match_panel(
            feat_rgb_i=np.asarray(feat_i, dtype=np.uint8),
            feat_rgb_j=np.asarray(feat_j, dtype=np.uint8),
            matches=pair_match_data[i],
            num_draw=int(args.num_matches_draw),
            line_width=int(args.line_width),
            point_radius=int(args.point_radius),
            draw_grid=bool(feature_draw_grid),
            vis_grid_h=feature_grid_h,
            vis_grid_w=feature_grid_w,
            match_grid_h=14,
            match_grid_w=14,
        )
        Image.fromarray(feat_match_panel).save(out_feat_match / f"pair_{i:03d}_feature_space_matching.png")

    # Curve plot.
    x = np.arange(len(pair_rows), dtype=np.int32)
    corr_scores = np.array([float(r["corr_score"]) for r in pair_rows], dtype=np.float32)
    match_means = np.array([float(r["match_mean_sim"]) for r in pair_rows], dtype=np.float32)
    fig = plt.figure(figsize=(12, 5), dpi=220)
    ax = fig.add_subplot(111)
    ax.plot(x, corr_scores, marker="o", linewidth=1.5, label="Adjacent Correspondence Score")
    ax.plot(x, match_means, marker="s", linewidth=1.2, label="Adjacent Match Mean Similarity")
    ax.set_xlabel("Adjacent Pair Index (t -> t+1)")
    ax.set_ylabel("Similarity")
    ax.set_title(f"Wan2.1-T2V Adjacent Consistency on {scene_id}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_root / "adjacent_consistency_curve.png")
    plt.close(fig)

    # Save tensors for further analysis.
    torch.save(feats14.to(torch.bfloat16), out_tensors / "wan_t2v_features_14x14.pt")
    torch.save(feats_prepool.to(torch.bfloat16), out_tensors / "wan_t2v_features_prepool.pt")
    torch.save(world14.to(torch.float32), out_tensors / "world_coords_14x14.pt")
    torch.save(valid14.to(torch.uint8), out_tensors / "valid_token_mask_14x14.pt")

    # Summary stats.
    def _safe_mean(vals: List[float]) -> float:
        vv = [float(v) for v in vals if math.isfinite(float(v))]
        return float(sum(vv) / len(vv)) if len(vv) > 0 else float("nan")

    summary = {
        "scene_id": scene_id,
        "split": args.split,
        "num_frames": int(args.num_frames),
        "num_adjacent_pairs": int(len(pair_rows)),
        "sampled_indices": [int(x) for x in sample_idx.tolist()],
        "corr_score_mean": _safe_mean([r["corr_score"] for r in pair_rows]),
        "corr_score_std": float(np.nanstd(np.array([r["corr_score"] for r in pair_rows], dtype=np.float32), ddof=1))
        if len(pair_rows) > 1
        else float("nan"),
        "match_mean_sim_mean": _safe_mean([r["match_mean_sim"] for r in pair_rows]),
        "match_num_selected_mean": _safe_mean([r["match_num_selected"] for r in pair_rows]),
        "match_mean_world_dist_mean": _safe_mean([r["match_mean_world_dist"] for r in pair_rows]),
        "config": {
            "checkpoint_dir": args.checkpoint_dir,
            "task": args.task,
            "size": args.size,
            "feat_block_idx": int(args.feat_block_idx),
            "timestep": int(args.timestep),
            "shift": float(args.shift),
            "dtype": args.dtype,
            "voxel_size": float(args.voxel_size),
            "valid_ratio_threshold": float(args.valid_ratio_threshold),
            "match_mutual_only": bool(mutual_only),
            "match_sim_threshold": float(args.match_sim_threshold),
            "match_max_world_dist": float(args.match_max_world_dist),
            "feature_map_scale": int(feature_map_scale),
            "feature_draw_grid": bool(feature_draw_grid),
            "feature_grid_h_prepool": int(feature_grid_h),
            "feature_grid_w_prepool": int(feature_grid_w),
            "feature_rgb_method": feature_rgb_info.get("method"),
        },
        "output_dir": str(out_root),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] scene={scene_id}, pairs={len(pair_rows)}")
    print(f"[done] corr_score_mean={summary['corr_score_mean']:.6f}, match_mean_sim_mean={summary['match_mean_sim_mean']:.6f}")
    print(f"[done] outputs saved to: {out_root}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
