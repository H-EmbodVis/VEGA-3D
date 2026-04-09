# Wan Visualization

This directory keeps simplified Wan visualization entrypoints.

Available commands:

- `bash scripts/visualization/run_wan_box_multiview.sh`
- `bash scripts/visualization/run_wan_correspondence.sh`
- `bash scripts/visualization/run_wan_adjacent.sh`
- `bash scripts/visualization/run_wan_triview_video.sh`

Examples:

```bash
bash scripts/visualization/run_wan_box_multiview.sh \
  --scene_id scannet/scene0000_00 \
  --object_id 39 \
  --box "-3.150150775909424,-2.9052600860595703,0.9401446161791682,0.60125732421875,1.2178735733032227,1.938113296404481"
```

```bash
bash scripts/visualization/run_wan_correspondence.sh \
  --scene_id scannet/scene0000_00 \
  --voxel_size 0.1
```

```bash
bash scripts/visualization/run_wan_adjacent.sh \
  --scene_id scannet/scene0000_00 \
  --match_sim_threshold 0.60
```

```bash
bash scripts/visualization/run_wan_triview_video.sh \
  --ego_video data/ego_view.mp4 \
  --left_video data/left_view.mp4 \
  --right_video data/right_view.mp4 \
  --sample_fps 10 \
  --output_fps 30
```

All four wrappers call `python3` directly and keep the heavier Wan runtime defaults inside the Python entrypoints.
