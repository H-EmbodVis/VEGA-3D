#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib

from _wan_vis_common import DEFAULT_VALID_RATIO_THRESHOLD, build_base_parser, build_runtime_namespace, default_output_dir


def parse_args() -> argparse.Namespace:
    parser = build_base_parser("Wan adjacent-frame correspondence and PCA feature visualization.")
    vis = parser.add_argument_group("Visualization")
    vis.add_argument("--voxel_size", type=float, default=0.1)
    vis.add_argument("--heatmap_alpha", type=float, default=0.55)
    vis.add_argument("--heatmap_gamma", type=float, default=1.2)
    vis.add_argument("--match_sim_threshold", type=float, default=0.60)
    vis.add_argument("--num_matches_draw", type=int, default=70)
    vis.add_argument("--feature_map_scale", type=int, default=32)
    vis.add_argument("--match_max_world_dist", type=float, default=0.0)
    vis.add_argument("--all_matches", dest="match_mutual_only", action="store_false")
    vis.add_argument("--no_feature_grid", dest="feature_draw_grid", action="store_false")
    parser.set_defaults(match_mutual_only=True, feature_draw_grid=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir("wan_adjacent", args.scene_id)

    impl = importlib.import_module("scripts.analysis.wan_t2v_adjacent_correspondence_matching")
    run_args = build_runtime_namespace(
        args,
        output_dir=output_dir,
        voxel_size=float(args.voxel_size),
        valid_ratio_threshold=DEFAULT_VALID_RATIO_THRESHOLD,
        heatmap_alpha=float(args.heatmap_alpha),
        norm_pmin=5.0,
        norm_pmax=95.0,
        heatmap_gamma=float(args.heatmap_gamma),
        match_mutual_only="true" if args.match_mutual_only else "false",
        match_sim_threshold=float(args.match_sim_threshold),
        match_max_world_dist=float(args.match_max_world_dist),
        num_matches_draw=int(args.num_matches_draw),
        line_width=2,
        point_radius=3,
        feature_map_scale=int(args.feature_map_scale),
        feature_draw_grid="true" if args.feature_draw_grid else "false",
    )
    impl.run(run_args)


if __name__ == "__main__":
    main()
