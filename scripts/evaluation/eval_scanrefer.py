"""
This script evaluates ScanRefer predictions by refining predicted bounding boxes
using precomputed proposals and calculating IoU metrics.
"""
import os
import ray
import json
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict
from lmms_eval.tasks.threedod.utils import EulerDepthInstance3DBoxes

def proposal_matching(proposals, pred_bbox):
    ret = pred_bbox
    max_iou = 0
    for proposal in proposals:
        cur_bbox = proposal + [0, 0, 0]
        try:
            iou = EulerDepthInstance3DBoxes.overlaps(
                EulerDepthInstance3DBoxes(torch.tensor([pred_bbox]), convention="ZXY"),
                EulerDepthInstance3DBoxes(torch.tensor([cur_bbox]), convention="ZXY")
            ).item()
        except Exception as e:
            print(f"Error in calculating IoU: {e}")
            continue
        if iou > max_iou:
            max_iou = iou
            ret = cur_bbox
    return ret

@ray.remote
def main(data, args):
    """
        The metadata file can be downloaded from the Video-3D-LLM data.
        https://huggingface.co/datasets/zd11024/Video-3D-LLM_data
    """
    with open(os.path.join(args.scannet_dir, "metadata", "scannet_val_pred_box.json")) as f:
        scan2obj = json.load(f)

    metrics = defaultdict(list)

    for item in tqdm(data):
        gt_bbox = item["doc"]["gt_bbox"]
        scene_id = item["scanrefer_score"]["images"][0].split("/")[-2]
        proposals = scan2obj[f"scannet/{scene_id}"]
        refined_bbox = proposal_matching(proposals, item["scanrefer_score"]["pred_bbox"])

        try:
            iou = EulerDepthInstance3DBoxes.overlaps(
                EulerDepthInstance3DBoxes(torch.tensor([refined_bbox]), convention="ZXY"),
                EulerDepthInstance3DBoxes(torch.tensor([gt_bbox]), convention="ZXY")
            ).item()
        except Exception as e:
            print(f"Error in calculating IoU: {e}")
            iou = 0
        
        metrics["overall_iou25"].append(iou >= 0.25)
        metrics["overall_iou50"].append(iou >= 0.5)

    return metrics

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="""Path to the ScanRefer prediction file in JSONL format. 
        You could produce the prdiction file by flagging --log_samples when running lmms-eval.""")
    parser.add_argument('--scannet_dir', type=str, default='data/media/scannet/')
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    with open(args.input_file) as f:
        data = [json.loads(line) for line in f.readlines()]

    ray.init()
    features = []
    for i in range(args.num_workers):
        features.append(main.remote(data[i::args.num_workers], args))
    
    metrics = defaultdict(list)
    for feature in features:
        feature = ray.get(feature)
        for k, v in feature.items():
            metrics[k].extend(v)

    overall_iou25 = sum(metrics["overall_iou25"]) / len(metrics["overall_iou25"]) * 100
    overall_iou50 = sum(metrics["overall_iou50"]) / len(metrics["overall_iou50"]) * 100

    print(f"Overall IoU@0.25: {overall_iou25:.2f}")
    print(f"Overall IoU@0.50: {overall_iou50:.2f}")