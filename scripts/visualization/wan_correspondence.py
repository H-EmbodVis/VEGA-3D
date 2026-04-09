#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib

from _wan_vis_common import DEFAULT_VALID_RATIO_THRESHOLD, build_base_parser, build_runtime_namespace, default_output_dir


def parse_args() -> argparse.Namespace:
    parser = build_base_parser("Wan scene-level correspondence-style multi-view consistency visualization.")
    vis = parser.add_argument_group("Visualization")
    vis.add_argument("--voxel_size", type=float, default=0.1)
    vis.add_argument("--overlay_mapping_mode", type=str, default="full_frame", choices=["full_frame", "strict_crop"])
    vis.add_argument("--heatmap_alpha", type=float, default=0.55)
    vis.add_argument("--heatmap_gamma", type=float, default=1.25)
    vis.add_argument("--component_top_quantile", type=float, default=85.0)
    vis.add_argument("--component_max_count", type=int, default=6)
    vis.add_argument("--component_alpha", type=float, default=0.68)
    vis.add_argument("--draw_topk_tokens", action="store_true")
    vis.add_argument("--token_topk_ratio", type=float, default=0.08)
    vis.add_argument("--no_component_coloring", dest="component_coloring", action="store_false")
    vis.add_argument("--no_save_component_ply", dest="save_component_ply", action="store_false")
    parser.set_defaults(component_coloring=True, save_component_ply=True, draw_topk_tokens=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir("wan_correspondence", args.scene_id)

    impl = importlib.import_module("scripts.analysis.wan_t2v_correspondence_multiview_vis")
    run_args = build_runtime_namespace(
        args,
        output_dir=output_dir,
        voxel_size=float(args.voxel_size),
        valid_ratio_threshold=DEFAULT_VALID_RATIO_THRESHOLD,
        overlay_mapping_mode=args.overlay_mapping_mode,
        heatmap_alpha=float(args.heatmap_alpha),
        norm_pmin=5.0,
        norm_pmax=95.0,
        heatmap_gamma=float(args.heatmap_gamma),
        component_coloring=bool(args.component_coloring),
        component_top_quantile=float(args.component_top_quantile),
        component_min_views=2,
        component_min_voxels=4,
        component_max_count=int(args.component_max_count),
        component_alpha=float(args.component_alpha),
        save_component_ply=bool(args.save_component_ply),
        draw_topk_tokens=bool(args.draw_topk_tokens),
        token_topk_ratio=float(args.token_topk_ratio),
    )
    impl.run(run_args)


if __name__ == "__main__":
    main()
