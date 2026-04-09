#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib

from _wan_vis_common import build_base_parser, build_runtime_namespace, default_output_dir


DEFAULT_BOX = "-3.150150775909424,-2.9052600860595703,0.9401446161791682,0.60125732421875,1.2178735733032227,1.938113296404481"


def parse_args() -> argparse.Namespace:
    parser = build_base_parser("Wan box-guided multi-view consistency visualization.")
    vis = parser.add_argument_group("Visualization")
    vis.add_argument("--box", type=str, default=DEFAULT_BOX, help="3D box in x,y,z,w,h,l format.")
    vis.add_argument("--object_id", type=int, default=39)
    vis.add_argument("--min_projected_area", type=float, default=16.0)
    vis.add_argument("--heatmap_alpha", type=float, default=0.55)
    vis.add_argument("--heatmap_gamma", type=float, default=1.4)
    vis.add_argument("--token_topk_ratio", type=float, default=0.08)
    vis.add_argument("--no_draw_topk_tokens", dest="draw_topk_tokens", action="store_false")
    vis.add_argument("--no_require_visible_instance", dest="require_visible_instance", action="store_false")
    parser.set_defaults(draw_topk_tokens=True, require_visible_instance=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir("wan_box_multiview", args.scene_id, object_id=args.object_id)

    impl = importlib.import_module("scripts.analysis.wan_t2v_box_multiview_consistency")
    run_args = build_runtime_namespace(
        args,
        output_dir=output_dir,
        box=args.box,
        object_id=int(args.object_id),
        description=getattr(impl, "DEFAULT_PROMPT", ""),
        require_visible_instance=bool(args.require_visible_instance),
        min_projected_area=float(args.min_projected_area),
        heatmap_alpha=float(args.heatmap_alpha),
        norm_pmin=60.0,
        norm_pmax=99.0,
        heatmap_gamma=float(args.heatmap_gamma),
        token_topk_ratio=float(args.token_topk_ratio),
        draw_topk_tokens=bool(args.draw_topk_tokens),
    )
    impl.run(run_args)


if __name__ == "__main__":
    main()
