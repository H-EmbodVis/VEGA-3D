#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import multiprocessing as mp
import os
import os.path as osp
import pickle
from typing import Dict, List, Optional, Tuple

import fire
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from llava.model.multimodal_generative_encoder.seva.geometry import get_plucker_coordinates, to_hom_pose
from llava.model.multimodal_generative_encoder.seva.model import SGMWrapper
from llava.model.multimodal_generative_encoder.seva.modules.autoencoder import AutoEncoder
from llava.model.multimodal_generative_encoder.seva.modules.conditioner import CLIPConditioner
from llava.model.multimodal_generative_encoder.seva.sampling import DiscreteDenoiser, MultiviewCFG
from llava.model.multimodal_generative_encoder.seva.utils import load_model_compatible


def _to_3x3_intrinsics(k: torch.Tensor) -> torch.Tensor:
    if k.shape[-2:] == (3, 3):
        return k
    if k.shape[-2:] == (4, 4):
        return k[..., :3, :3]
    raise ValueError(f"Unexpected intrinsic shape: {tuple(k.shape)}")


def _to_3x4_pose(pose: torch.Tensor) -> torch.Tensor:
    if pose.shape[-2:] == (3, 4):
        return pose
    if pose.shape[-2:] == (4, 4):
        return pose[..., :3, :]
    raise ValueError(f"Unexpected pose shape: {tuple(pose.shape)}")


def _resize_with_intrinsics(
    frames: torch.Tensor,
    intrinsics: torch.Tensor,
    out_h: int,
    out_w: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n, _, h, w = frames.shape
    scale = max(out_h / h, out_w / w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    x = F.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False)
    top = max(0, (new_h - out_h) // 2)
    left = max(0, (new_w - out_w) // 2)
    x = x[:, :, top : top + out_h, left : left + out_w]

    k = intrinsics.clone()
    k[:, 0, :] *= float(new_w) / float(w)
    k[:, 1, :] *= float(new_h) / float(h)
    k[:, 0, 2] -= left
    k[:, 1, 2] -= top
    return x, k


def _load_scene_items(annotation_dir: str) -> List[Tuple[str, Dict]]:
    scene_dict_all = {}
    for split in ["train", "val", "test"]:
        pkl_path = osp.join(annotation_dir, f"embodiedscan_infos_{split}.pkl")
        if not osp.exists(pkl_path):
            print(f"Warning: {pkl_path} not found, skipping split {split}")
            continue
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)["data_list"]
            for item in data:
                if item["sample_idx"].startswith("scannet"):
                    scene_dict_all[item["sample_idx"]] = item
    return [(video_id, scene_dict) for video_id, scene_dict in scene_dict_all.items()]


def _sample_frame_indices(total_frames: int, num_frames: int) -> List[int]:
    if total_frames <= 0:
        raise ValueError("No frames found.")
    if total_frames == 1:
        return [0] * num_frames
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    return [max(0, min(total_frames - 1, idx)) for idx in indices]


def _load_rgb_depth_pose_intrinsics(
    scene_dict: Dict,
    video_folder: str,
    num_frames: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    all_frame_files = [osp.join(video_folder, img["img_path"]) for img in scene_dict["images"]]
    selected_indices = _sample_frame_indices(len(all_frame_files), num_frames)
    frame_files = [all_frame_files[i] for i in selected_indices]

    cam2img = np.array(scene_dict["cam2img"], dtype=np.float32)
    if cam2img.shape == (4, 4):
        k_3x3 = cam2img[:3, :3]
    elif cam2img.shape == (3, 3):
        k_3x3 = cam2img
    else:
        raise ValueError(f"Unexpected cam2img shape: {cam2img.shape}")

    axis_align_matrix = np.array(scene_dict["axis_align_matrix"], dtype=np.float32)
    frames = []
    c2ws = []
    depths = []
    has_all_depth = True

    for frame_path in frame_files:
        with Image.open(frame_path) as img:
            arr = np.array(img.convert("RGB"), copy=True)
        frames.append(torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0)

        pose_path = frame_path.replace(".jpg", ".txt")
        if not osp.exists(pose_path):
            pose_path = osp.splitext(frame_path)[0] + ".txt"
        if not osp.exists(pose_path):
            raise FileNotFoundError(f"Pose file not found: {pose_path}")
        pose = np.loadtxt(pose_path).reshape(4, 4).astype(np.float32)
        pose = axis_align_matrix @ pose
        c2ws.append(torch.from_numpy(pose[:3, :]))

        depth_path = frame_path.replace(".jpg", ".png")
        if osp.exists(depth_path):
            with Image.open(depth_path) as depth_img:
                depth_np = np.array(depth_img, copy=True).astype(np.float32)
            depths.append(torch.from_numpy(depth_np))
        else:
            has_all_depth = False

    x = torch.stack(frames, dim=0) * 2.0 - 1.0
    c2w = torch.stack(c2ws, dim=0).float()
    k = torch.from_numpy(np.stack([k_3x3] * len(frame_files), axis=0)).float()
    depth_tensor = torch.stack(depths, dim=0) if has_all_depth and len(depths) == len(frame_files) else None
    return x, c2w, k, depth_tensor


def _build_value_dict(
    imgs: torch.Tensor,
    input_indices: List[int],
    c2ws: torch.Tensor,
    ks: torch.Tensor,
    camera_scale: float,
) -> Dict[str, torch.Tensor]:
    t = imgs.shape[0]
    h, w = imgs.shape[-2:]
    f = 8
    value_dict: Dict[str, torch.Tensor] = {}
    value_dict["cond_frames"] = imgs
    value_dict["cond_frames_mask"] = torch.zeros(t, dtype=torch.bool, device=imgs.device)
    value_dict["cond_frames_mask"][input_indices] = True

    c2w = to_hom_pose(c2ws.float())
    w2c = torch.linalg.inv(c2w)
    camera_dist = torch.norm(c2w[:, :3, 3] - c2w[:, :3, 3].median(0, keepdim=True).values, dim=-1)
    valid_mask = camera_dist <= torch.clamp(torch.quantile(camera_dist, 0.97) * 10, max=1e6)
    if valid_mask.any():
        c2w[:, :3, 3] -= c2w[valid_mask, :3, 3].mean(0, keepdim=True)
    else:
        c2w[:, :3, 3] -= c2w[:, :3, 3].mean(0, keepdim=True)
    w2c = torch.linalg.inv(c2w)

    camera_dists = c2w[:, :3, 3].clone()
    first_norm = torch.norm(camera_dists[0]).clamp(min=1e-6)
    if torch.isclose(first_norm, torch.zeros(1, device=imgs.device), atol=1e-5).any():
        translation_scaling_factor = camera_scale
    else:
        translation_scaling_factor = camera_scale / first_norm
    w2c[:, :3, 3] *= translation_scaling_factor
    c2w[:, :3, 3] *= translation_scaling_factor

    value_dict["plucker_coordinate"] = get_plucker_coordinates(
        extrinsics_src=w2c[0],
        extrinsics=w2c,
        intrinsics=ks.float().clone(),
        target_size=(h // f, w // f),
    )
    value_dict["c2w"] = c2w
    value_dict["K"] = ks
    return value_dict


def _extract_scene_feature(
    model: SGMWrapper,
    ae: AutoEncoder,
    conditioner: CLIPConditioner,
    denoiser: DiscreteDenoiser,
    guider: MultiviewCFG,
    scene_dict: Dict,
    video_folder: str,
    num_frames: int,
    input_size: int,
    output_spatial: int,
    timestep: int,
    cfg_scale: float,
    camera_scale: float,
    param_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    imgs, c2ws, ks, depths = _load_rgb_depth_pose_intrinsics(scene_dict, video_folder, num_frames)
    k = _to_3x3_intrinsics(ks)
    c2w = _to_3x4_pose(c2ws)
    imgs, k = _resize_with_intrinsics(imgs, k, input_size, input_size)

    imgs = imgs.to(device=device)
    c2w = c2w.to(device=device)
    k = k.to(device=device)
    if depths is not None:
        depths = (
            F.interpolate(
                depths.unsqueeze(1),
                size=(input_size, input_size),
                mode="nearest",
            )
            .squeeze(1)
            .to(device=device)
        )

    t = imgs.shape[0]
    num_inputs = max(1, t // 2)
    if depths is not None:
        valid_ratio = (depths > 0).float().flatten(1).mean(dim=1)
        input_indices = torch.topk(valid_ratio, k=num_inputs, largest=True).indices.sort().values.tolist()
    else:
        input_indices = torch.linspace(0, t - 1, num_inputs, device=device).round().long().tolist()

    value_dict = _build_value_dict(
        imgs=imgs,
        input_indices=input_indices,
        c2ws=c2w,
        ks=k,
        camera_scale=camera_scale,
    )
    input_mask = value_dict["cond_frames_mask"]

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=param_dtype):
        latents = F.pad(ae.encode(imgs[input_mask], 1), (0, 0, 0, 0, 0, 1), value=1.0)
        text_emb = conditioner(imgs[input_mask]).mean(0, keepdim=True)
        c_crossattn = text_emb.unsqueeze(1).expand(t, 1, -1).contiguous()
        uc_crossattn = torch.zeros_like(c_crossattn)

        c_replace = latents.new_zeros(t, *latents.shape[1:])
        c_replace[input_mask] = latents
        uc_replace = torch.zeros_like(c_replace)
        plucker = value_dict["plucker_coordinate"]
        c_concat = torch.cat(
            [input_mask[:, None, None, None].expand(-1, 1, plucker.shape[-2], plucker.shape[-1]).to(plucker.dtype), plucker],
            dim=1,
        )
        uc_concat = torch.cat([plucker.new_zeros(t, 1, *plucker.shape[-2:]), plucker], dim=1)
        cond = {
            "crossattn": c_crossattn,
            "replace": c_replace,
            "concat": c_concat,
            "dense_vector": plucker,
        }
        uc = {
            "crossattn": uc_crossattn,
            "replace": uc_replace,
            "concat": uc_concat,
            "dense_vector": plucker,
        }

        x0 = ae.encode(imgs, 1)
        sigma_idx = max(0, min(denoiser.sigmas.shape[0] - 1, int(timestep)))
        sigma = denoiser.idx_to_sigma(torch.tensor(sigma_idx, device=device, dtype=torch.long)).to(x0.dtype)
        s = sigma.expand(t)
        noise = torch.randn_like(x0)
        x_noisy = x0 + noise * sigma.view(1, 1, 1, 1)

        x_in, s_in, c_in = guider.prepare_inputs(x_noisy, s, cond, uc)
        denoised = denoiser(model, x_in, s_in, c_in, num_frames=t)
        _ = guider(
            denoised,
            s,
            scale=cfg_scale,
            c2w=value_dict["c2w"],
            K=value_dict["K"],
            input_frame_mask=input_mask,
        )

        feat = model.module.last_input_feat
        if feat is None:
            raise RuntimeError("Failed to capture intermediate feature from model.module.last_input_feat.")
        feat = feat.to(device=device, dtype=param_dtype)
        if feat.shape[0] % 2 == 0:
            feat = feat[feat.shape[0] // 2 :]
        if feat.shape[-2] != output_spatial or feat.shape[-1] != output_spatial:
            feat = F.adaptive_avg_pool2d(feat, output_size=(output_spatial, output_spatial))
    return feat.detach().to(dtype=param_dtype).cpu()


def _process_scenes_on_gpu(
    scene_items: List[Tuple[str, Dict]],
    device_id: int,
    model_type: str,
    checkpoint_dir: str,
    weight_name: str,
    model_version: float,
    sd_model_path: str,
    clip_model_path: str,
    save_root: str,
    save_subdir: str,
    video_folder: str,
    num_frames: int,
    input_size: int,
    output_spatial: int,
    timestep: int,
    cfg_scale: float,
    cfg_min: float,
    camera_scale: float,
    skip_saved: bool,
) -> None:
    if len(scene_items) == 0:
        return

    device = torch.device(f"cuda:{device_id}")
    param_dtype = torch.bfloat16
    print(f"[GPU {device_id}] loading {model_type} modules from {checkpoint_dir} ...")

    ae = AutoEncoder(chunk_size=1, sd_model_path=sd_model_path).eval().requires_grad_(False)
    conditioner = CLIPConditioner(pretrained_path=clip_model_path).eval().requires_grad_(False)
    denoiser = DiscreteDenoiser(num_idx=1000, device=device)
    guider = MultiviewCFG(cfg_min=cfg_min)

    module = load_model_compatible(
        pretrained_model_name_or_path=checkpoint_dir,
        weight_name=weight_name,
        model_version=model_version,
        device="cpu",
        verbose=True,
    ).eval()
    model = SGMWrapper(module).eval().requires_grad_(False)

    ae.to(device=device, dtype=param_dtype)
    conditioner.to(device=device, dtype=param_dtype)
    model.to(device=device, dtype=param_dtype)

    for video_id, scene_dict in scene_items:
        scene_name = video_id.split("/")[-1]
        save_path_scene = osp.join(save_root, save_subdir, scene_name)
        os.makedirs(save_path_scene, exist_ok=True)
        feature_path = osp.join(save_path_scene, "seva.pt")
        if skip_saved and osp.exists(feature_path):
            print(f"[GPU {device_id}] skip {scene_name} (exists)")
            continue

        try:
            feat = _extract_scene_feature(
                model=model,
                ae=ae,
                conditioner=conditioner,
                denoiser=denoiser,
                guider=guider,
                scene_dict=scene_dict,
                video_folder=video_folder,
                num_frames=num_frames,
                input_size=input_size,
                output_spatial=output_spatial,
                timestep=timestep,
                cfg_scale=cfg_scale,
                camera_scale=camera_scale,
                param_dtype=param_dtype,
                device=device,
            )
            torch.save(feat, feature_path)
            print(f"[GPU {device_id}] saved {scene_name}: {tuple(feat.shape)}")
        except Exception as exc:
            print(f"[GPU {device_id}] failed {scene_name}: {exc}")
        finally:
            torch.cuda.empty_cache()


def main(
    model_type: str = "seva",
    annotation_dir: str = "data/embodiedscan/",
    video_folder: str = "data",
    save_root: str = "data/scannet",
    save_subdir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    weight_name: Optional[str] = None,
    model_version: Optional[float] = None,
    sd_model_path: str = "data/models/stable-diffusion-2-1-base",
    clip_model_path: str = "data/models/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_model.safetensors",
    num_frames: int = 32,
    input_size: int = 896,
    output_spatial: int = 14,
    timestep: int = 250,
    cfg_scale: float = 2.0,
    cfg_min: float = 1.2,
    camera_scale: float = 2.0,
    num_gpus: int = 8,
    skip_saved: bool = True,
    data_items: Optional[str] = None,
) -> None:
    model_type = str(model_type).lower().strip()
    if model_type not in {"seva", "vmem"}:
        raise ValueError("model_type must be one of: seva, vmem")

    if save_subdir is None:
        save_subdir = (
            "uniform_seva_middle_feats_onestep_t250_input16_patch14"
            if model_type == "seva"
            else "uniform_vmem_middle_feats_onestep_t250_input16_patch14"
        )
    if checkpoint_dir is None:
        checkpoint_dir = (
            "data/models/stable-virtual-camera"
            if model_type == "seva"
            else "data/models/vmem"
        )
    if weight_name is None:
        weight_name = "modelv1.1.safetensors" if model_type == "seva" else "vmem_weights.pth"
    if model_version is None:
        model_version = 1.1 if model_type == "seva" else 1.0

    scene_items = _load_scene_items(annotation_dir)
    if len(scene_items) == 0:
        raise RuntimeError("No scenes found in embodiedscan pkl files.")

    if data_items is not None:
        wanted = set([x.strip() for x in str(data_items).split(",") if x.strip()])

        def _match(video_id: str) -> bool:
            scene_name = video_id.split("/")[-1]
            if video_id in wanted or scene_name in wanted:
                return True
            return any(scene_name.endswith(item) for item in wanted)

        scene_items = [(vid, item) for vid, item in scene_items if _match(vid)]

    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise RuntimeError("No CUDA devices available.")
    if num_gpus > available_gpus:
        print(f"Requested num_gpus={num_gpus} but only {available_gpus} visible; clamping.")
        num_gpus = available_gpus

    print(
        f"model_type={model_type} scenes={len(scene_items)} num_gpus={num_gpus} "
        f"num_frames={num_frames} input_size={input_size} output_spatial={output_spatial} timestep={timestep}"
    )
    print(f"checkpoint_dir={checkpoint_dir} weight_name={weight_name} save_dir={osp.join(save_root, save_subdir)}")

    scenes_per_gpu = [scene_items[i::num_gpus] for i in range(num_gpus)]
    tasks = []
    for gpu_id in range(num_gpus):
        if len(scenes_per_gpu[gpu_id]) == 0:
            continue
        tasks.append(
            (
                scenes_per_gpu[gpu_id],
                gpu_id,
                model_type,
                checkpoint_dir,
                weight_name,
                float(model_version),
                sd_model_path,
                clip_model_path,
                save_root,
                save_subdir,
                video_folder,
                int(num_frames),
                int(input_size),
                int(output_spatial),
                int(timestep),
                float(cfg_scale),
                float(cfg_min),
                float(camera_scale),
                bool(skip_saved),
            )
        )

    if len(tasks) > 0:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(tasks)) as pool:
            pool.starmap(_process_scenes_on_gpu, tasks)

    print("all scenes done")


if __name__ == "__main__":
    fire.Fire(main)
