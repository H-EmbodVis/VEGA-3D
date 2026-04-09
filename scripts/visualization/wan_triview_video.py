#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import av
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from _wan_vis_common import (
    DEFAULT_BLOCK_IDX,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_SEED,
    DEFAULT_SHIFT,
    DEFAULT_SIZE,
    DEFAULT_TASK,
    DEFAULT_TIMESTEP,
)


VIEW_NAMES = ("ego", "left", "right")
VIEW_COLORS = {
    "ego": (255, 120, 120),
    "left": (120, 220, 120),
    "right": (120, 180, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample synchronized ego/left/right videos, extract Wan generative features, and render a tri-view PCA visualization video."
    )
    parser.add_argument("--ego_video", type=str, default="data/ego_view.mp4")
    parser.add_argument("--left_video", type=str, default="data/left_view.mp4")
    parser.add_argument("--right_video", type=str, default="data/right_view.mp4")
    parser.add_argument("--output_dir", type=str, default="results/visualization/wan_triview_video")
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--sample_fps", type=float, default=10.0, help="FPS used for Wan feature extraction.")
    parser.add_argument("--output_fps", type=float, default=30.0, help="FPS of the rendered output video.")
    parser.add_argument("--max_seconds", type=float, default=0.0, help="<=0 means use the full common duration.")
    parser.add_argument("--block_idx", type=int, default=DEFAULT_BLOCK_IDX)
    parser.add_argument("--timestep", type=int, default=DEFAULT_TIMESTEP)
    parser.add_argument("--size", type=str, default=DEFAULT_SIZE)
    parser.add_argument("--dtype", type=str, default=DEFAULT_DTYPE, choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--display_height", type=int, default=240)
    parser.add_argument("--overlay_alpha", type=float, default=0.35)
    parser.add_argument("--title", type=str, default="Wan Generative Features: Tri-View PCA Consistency")
    return parser.parse_args()


def _video_times(path: str) -> Tuple[np.ndarray, float, int]:
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    times: List[float] = []
    for idx, frame in enumerate(container.decode(stream)):
        if frame.time is not None:
            times.append(float(frame.time))
        else:
            fps = float(stream.average_rate) if stream.average_rate is not None else 0.0
            times.append(float(idx) / fps if fps > 0 else float(idx))
    fps = float(stream.average_rate) if stream.average_rate is not None else 0.0
    container.close()
    return np.asarray(times, dtype=np.float64), fps, len(times)


def _nearest_indices(times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(times, target_times, side="left")
    idx = np.clip(idx, 0, max(0, len(times) - 1))
    prev = np.clip(idx - 1, 0, max(0, len(times) - 1))
    choose_prev = np.abs(target_times - times[prev]) <= np.abs(times[idx] - target_times)
    return np.where(choose_prev, prev, idx).astype(np.int64)


def _build_encoder(args: argparse.Namespace) -> Any:
    from llava.model.multimodal_generative_encoder.wan_t2v_encoder import WanT2VOnlineEncoder

    cfg = SimpleNamespace(
        generative_vision_tower_task=DEFAULT_TASK,
        generative_vision_tower_checkpoint=args.checkpoint_dir,
        generative_vision_tower_size=args.size,
        generative_vision_tower_timestep=int(args.timestep),
        generative_vision_tower_shift=float(DEFAULT_SHIFT),
        generative_vision_tower_feat_block_idx=int(args.block_idx),
        generative_vision_tower_dtype=args.dtype,
    )
    return WanT2VOnlineEncoder(cfg).eval()


def _to_neg_one_to_one(frames: torch.Tensor) -> torch.Tensor:
    return frames.float() * 2.0 - 1.0


def _resize_center_crop(frames: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    if frames.ndim != 4:
        raise ValueError(f"Expected [N, C, H, W], got {tuple(frames.shape)}")
    scale = max(out_h / float(frames.shape[-2]), out_w / float(frames.shape[-1]))
    new_h = max(1, int(round(frames.shape[-2] * scale)))
    new_w = max(1, int(round(frames.shape[-1] * scale)))
    x = F.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False)
    top = max(0, (new_h - out_h) // 2)
    left = max(0, (new_w - out_w) // 2)
    return x[:, :, top : top + out_h, left : left + out_w]


def _encode_prepool_chunk(
    encoder: Any,
    frames_unit: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    encoder._move_models_to_device(device)
    use_autocast = device.type == "cuda" and encoder.param_dtype in (torch.float16, torch.bfloat16)

    x = frames_unit.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
    x_proc = encoder._prepare_frames(x)
    frame_list = [x_proc[i].unsqueeze(1) for i in range(x_proc.shape[0])]

    encoder._ensure_scheduler_ready(device)
    tau = encoder._select_timestep(encoder.scheduler.timesteps, target_timestep=int(encoder.timestep))
    context = encoder._get_text_context(device=device, batch_size=len(frame_list))
    block_idx = (len(encoder.model.blocks) - 1) if encoder.feat_block_idx < 0 else encoder.feat_block_idx
    if block_idx < 0 or block_idx >= len(encoder.model.blocks):
        raise ValueError(f"feat_block_idx out of range: {block_idx}")

    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=encoder.param_dtype, enabled=use_autocast):
            base_latents = encoder.vae.encode(frame_list)
            target_shape = list(base_latents[0].shape)
            seq_len = int(np.ceil((target_shape[2] * target_shape[3]) / (encoder.patch_size[1] * encoder.patch_size[2]) * target_shape[1]))

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
                _ = encoder.model(noisy_latents_list, t=t, context=context, seq_len=seq_len)
            finally:
                handle.remove()

            feats = feat_holder.get("feat", None)
            if feats is None:
                raise RuntimeError("Failed to capture WAN-T2V pre-pooling features.")

            grid_h = encoder.frame_height // (encoder.vae_stride[1] * encoder.patch_size[1])
            grid_w = encoder.frame_width // (encoder.vae_stride[2] * encoder.patch_size[2])
            tokens_per_frame = grid_h * grid_w
            if feats.shape[1] != tokens_per_frame:
                raise RuntimeError(f"Unexpected token count: {feats.shape[1]} (expected {tokens_per_frame}).")
            feats = feats.view(feats.shape[0], grid_h, grid_w, feats.shape[2]).permute(0, 3, 1, 2).contiguous()

    out = feats.float().cpu()
    del x, x_proc, frame_list, base_latents, latent_batch, noise, noisy_latents, noisy_latents_list, feats
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def _prepare_display_frames(frames_unit: torch.Tensor, out_h: int, out_w: int, display_h: int, display_w: int) -> np.ndarray:
    proc = _resize_center_crop(_to_neg_one_to_one(frames_unit), out_h=out_h, out_w=out_w)
    proc = ((proc + 1.0) * 0.5).clamp(0.0, 1.0)
    proc = F.interpolate(proc, size=(display_h, display_w), mode="bilinear", align_corners=False)
    return (proc.mul(255.0).add_(0.5).clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy())


def _process_selected_video(
    path: str,
    selected_indices: np.ndarray,
    encoder: Any,
    device: torch.device,
    wan_frame_h: int,
    wan_frame_w: int,
    display_h: int,
    display_w: int,
    chunk_size: int,
) -> Tuple[np.ndarray, torch.Tensor]:
    unique_indices, inverse = np.unique(selected_indices.astype(np.int64), return_inverse=True)
    unique_display: List[np.ndarray] = []
    unique_feats: List[torch.Tensor] = []

    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    batch_frames: List[np.ndarray] = []
    target_ptr = 0

    def _flush_batch() -> None:
        nonlocal batch_frames
        if not batch_frames:
            return
        chunk_np = np.stack(batch_frames, axis=0)
        frames_unit = torch.from_numpy(chunk_np).permute(0, 3, 1, 2).float() / 255.0
        unique_display.append(_prepare_display_frames(frames_unit, out_h=wan_frame_h, out_w=wan_frame_w, display_h=display_h, display_w=display_w))
        unique_feats.append(_encode_prepool_chunk(encoder, frames_unit, device=device))
        batch_frames = []

    for idx, frame in enumerate(container.decode(stream)):
        if target_ptr >= len(unique_indices):
            break
        if idx != int(unique_indices[target_ptr]):
            continue
        batch_frames.append(frame.to_ndarray(format="rgb24"))
        target_ptr += 1
        if len(batch_frames) >= int(chunk_size):
            _flush_batch()
    _flush_batch()
    container.close()

    if target_ptr != len(unique_indices):
        raise RuntimeError(f"Failed to decode all requested frames from {path}. decoded={target_ptr}, expected={len(unique_indices)}")

    unique_display_np = np.concatenate(unique_display, axis=0)
    unique_feats_t = torch.cat(unique_feats, dim=0)
    return unique_display_np[inverse], unique_feats_t[inverse]


def _pair_token_consistency(feat_a: torch.Tensor, feat_b: torch.Tensor) -> float:
    a = F.normalize(feat_a.permute(1, 2, 0).reshape(-1, feat_a.shape[0]).float(), dim=-1)
    b = F.normalize(feat_b.permute(1, 2, 0).reshape(-1, feat_b.shape[0]).float(), dim=-1)
    sim = a @ b.t()
    score = 0.5 * (sim.max(dim=1).values.mean() + sim.max(dim=0).values.mean())
    return float(score.item())


def _resize_rgb(rgb: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    return np.asarray(Image.fromarray(rgb).resize((out_w, out_h), resample=Image.NEAREST), dtype=np.uint8)


def _overlay(base_rgb: np.ndarray, feat_rgb: np.ndarray, alpha: float) -> np.ndarray:
    base = base_rgb.astype(np.float32)
    feat = feat_rgb.astype(np.float32)
    out = (1.0 - float(alpha)) * base + float(alpha) * feat
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_tile_label(tile: np.ndarray, text: str, color: Tuple[int, int, int]) -> np.ndarray:
    img = Image.fromarray(tile)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, min(img.width - 1, 180), 26], fill=(0, 0, 0))
    draw.text((8, 6), text, fill=color)
    return np.asarray(img, dtype=np.uint8)


def _make_footer(width: int, height: int, title: str, line: str) -> np.ndarray:
    img = Image.new("RGB", (width, height), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)
    draw.text((10, 6), title, fill=(255, 255, 255))
    draw.text((10, 28), line, fill=(210, 210, 210))
    return np.asarray(img, dtype=np.uint8)


def _write_video(path: Path, frames: Sequence[np.ndarray], fps: float) -> None:
    if len(frames) == 0:
        raise ValueError("No frames to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=max(1, int(round(float(fps)))))
    stream.width = int(frames[0].shape[1])
    stream.height = int(frames[0].shape[0])
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "medium"}

    for frame in frames:
        video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in stream.encode(video_frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    try:
        from llava.model.multimodal_generative_encoder.wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES
        from scripts.visualization.wan_t2v_adjacent_correspondence_matching import _compute_feature_rgb_maps
    except Exception as exc:
        raise RuntimeError(
            "Failed to import WAN visualization dependencies. "
            "Check the local environment, especially transformers / huggingface-hub compatibility."
        ) from exc

    if args.size not in SIZE_CONFIGS:
        raise ValueError(f"Unsupported size: {args.size}")
    if DEFAULT_TASK not in SUPPORTED_SIZES or args.size not in SUPPORTED_SIZES[DEFAULT_TASK]:
        raise ValueError(f"Size `{args.size}` not supported by task `{DEFAULT_TASK}`: {SUPPORTED_SIZES.get(DEFAULT_TASK)}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but --device requests CUDA.")

    video_paths = {
        "ego": args.ego_video,
        "left": args.left_video,
        "right": args.right_video,
    }
    video_meta = {}
    min_duration = None
    min_source_fps = None
    for name, path in video_paths.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"Video not found: {path}")
        times, fps, frame_count = _video_times(path)
        if times.size == 0:
            raise RuntimeError(f"No frames decoded from {path}")
        duration = float(times[-1])
        video_meta[name] = {"times": times, "fps": fps, "frame_count": frame_count, "duration": duration}
        min_duration = duration if min_duration is None else min(min_duration, duration)
        min_source_fps = fps if min_source_fps is None else min(min_source_fps, fps)

    if float(args.sample_fps) <= 0 or float(args.output_fps) <= 0:
        raise ValueError("sample_fps and output_fps must be > 0.")
    if min_source_fps is not None and float(args.sample_fps) > float(min_source_fps) + 1e-6:
        raise ValueError(f"sample_fps={args.sample_fps} exceeds the minimum source fps={min_source_fps:.3f}. Use output_fps for smoother rendering.")

    common_duration = float(min_duration)
    if float(args.max_seconds) > 0:
        common_duration = min(common_duration, float(args.max_seconds))
    sample_times = np.arange(0.0, common_duration + 1e-8, 1.0 / float(args.sample_fps), dtype=np.float64)
    output_times = np.arange(0.0, common_duration + 1e-8, 1.0 / float(args.output_fps), dtype=np.float64)
    if sample_times.size == 0 or output_times.size == 0:
        raise RuntimeError("No synchronized timestamps were generated.")

    encoder = _build_encoder(args)
    frame_w, frame_h = SIZE_CONFIGS[args.size]
    display_h = int(args.display_height)
    display_w = int(round(frame_w * (display_h / float(frame_h))))

    view_display: Dict[str, np.ndarray] = {}
    view_feats: Dict[str, torch.Tensor] = {}
    for name in VIEW_NAMES:
        selected_indices = _nearest_indices(video_meta[name]["times"], sample_times)
        display_np, feats_t = _process_selected_video(
            path=video_paths[name],
            selected_indices=selected_indices,
            encoder=encoder,
            device=device,
            wan_frame_h=frame_h,
            wan_frame_w=frame_w,
            display_h=display_h,
            display_w=display_w,
            chunk_size=int(args.chunk_size),
        )
        view_display[name] = display_np
        view_feats[name] = feats_t

    feats_all = torch.stack([view_feats[name] for name in VIEW_NAMES], dim=1)  # [T,3,C,H,W]
    t, v, c, hp, wp = feats_all.shape
    feat_rgb_flat, pca_info = _compute_feature_rgb_maps(feats_all.permute(0, 1, 3, 4, 2).reshape(t * v, hp, wp, c))
    feat_rgb = feat_rgb_flat.reshape(t, v, hp, wp, 3)

    feats14 = F.adaptive_avg_pool2d(feats_all.reshape(t * v, c, hp, wp), output_size=(14, 14)).reshape(t, v, c, 14, 14)
    pair_scores = {
        "ego_left": [],
        "ego_right": [],
        "left_right": [],
    }
    for i in range(t):
        pair_scores["ego_left"].append(_pair_token_consistency(feats14[i, 0], feats14[i, 1]))
        pair_scores["ego_right"].append(_pair_token_consistency(feats14[i, 0], feats14[i, 2]))
        pair_scores["left_right"].append(_pair_token_consistency(feats14[i, 1], feats14[i, 2]))

    panels: List[np.ndarray] = []
    footer_h = 54
    for i in range(t):
        raw_tiles = []
        pca_tiles = []
        overlay_tiles = []
        for view_idx, view_name in enumerate(VIEW_NAMES):
            raw = view_display[view_name][i]
            pca = _resize_rgb(feat_rgb[i, view_idx], out_h=display_h, out_w=display_w)
            overlay = _overlay(raw, pca, alpha=float(args.overlay_alpha))
            raw_tiles.append(_draw_tile_label(raw, f"{view_name} input", VIEW_COLORS[view_name]))
            pca_tiles.append(_draw_tile_label(pca, f"{view_name} PCA", VIEW_COLORS[view_name]))
            overlay_tiles.append(_draw_tile_label(overlay, f"{view_name} overlay", VIEW_COLORS[view_name]))

        row_raw = np.concatenate(raw_tiles, axis=1)
        row_pca = np.concatenate(pca_tiles, axis=1)
        row_overlay = np.concatenate(overlay_tiles, axis=1)
        footer = _make_footer(
            width=row_raw.shape[1],
            height=footer_h,
            title=args.title,
            line=(
                f"t={sample_times[i]:05.2f}s | token consistency: "
                f"ego-left {pair_scores['ego_left'][i]:.3f} | "
                f"ego-right {pair_scores['ego_right'][i]:.3f} | "
                f"left-right {pair_scores['left_right'][i]:.3f}"
            ),
        )
        panels.append(np.concatenate([row_raw, row_pca, row_overlay, footer], axis=0))

    output_frame_indices = _nearest_indices(sample_times, output_times)
    output_frames = [panels[int(i)] for i in output_frame_indices.tolist()]

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    preview_dir = out_root / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panels[0]).save(preview_dir / "panel_first.png")
    Image.fromarray(panels[len(panels) // 2]).save(preview_dir / "panel_middle.png")
    Image.fromarray(panels[-1]).save(preview_dir / "panel_last.png")

    video_path = out_root / "wan_triview_pca_consistency.mp4"
    _write_video(video_path, output_frames, fps=float(args.output_fps))

    summary = {
        "videos": video_paths,
        "size": args.size,
        "checkpoint_dir": args.checkpoint_dir,
        "device": args.device,
        "sample_fps": float(args.sample_fps),
        "output_fps": float(args.output_fps),
        "common_duration": float(common_duration),
        "num_sample_frames": int(sample_times.size),
        "num_output_frames": int(len(output_frames)),
        "display_size": {"width": int(display_w), "height": int(display_h)},
        "pca_info": pca_info,
        "pair_scores_mean": {k: float(np.mean(v)) for k, v in pair_scores.items()},
        "pair_scores_std": {k: float(np.std(v)) for k, v in pair_scores.items()},
        "pair_scores_per_sample": pair_scores,
        "sample_times": [float(x) for x in sample_times.tolist()],
        "output_video": str(video_path),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] sample_frames={sample_times.size}, output_frames={len(output_frames)}, duration={common_duration:.2f}s")
    print(f"[done] pair score means: ego-left={summary['pair_scores_mean']['ego_left']:.4f}, ego-right={summary['pair_scores_mean']['ego_right']:.4f}, left-right={summary['pair_scores_mean']['left_right']:.4f}")
    print(f"[done] video saved to: {video_path}")


if __name__ == "__main__":
    main()
