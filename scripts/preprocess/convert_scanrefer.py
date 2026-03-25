import os
import csv
import ray
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import pickle
from lmms_eval.tasks.threedod.utils import EulerDepthInstance3DBoxes
from utils import _9dof_transform_world2cam, sample_images_and_best_view, uniform_sample_images

# modified from https://github.com/3dlg-hcvc/M3DRef-CLIP/blob/main/dataset/scanrefer/add_evaluation_labels.py
def get_semantic_mapping_file(file_path, mapping_name):
    label_mapping = {}
    mapping_col_idx = {
        "nyu40": 4,
        "eigen13": 5,
        "mpcat40": 16
    }
    with open(file_path, "r") as f:
        tsv_file = csv.reader(f, delimiter="\t")
        next(tsv_file)  # skip the header
        for line in tsv_file:
            label_mapping[line[1]] = int(line[mapping_col_idx[mapping_name]])
    return label_mapping


def add_unique_multiple_labels_to_json(file_path, label_mapping, valid_semantic_mapping):
    with open(file_path, "r") as f:
        scanrefer_json_data = json.load(f)
    obj_cache = {}
    sem_cache = {}
    for item in scanrefer_json_data:
        if (item["scene_id"], item["object_id"]) in obj_cache:
            continue
        obj_name = item["object_name"].replace("_", " ")
        sem_label = 39
        if obj_name in label_mapping:
            sem_label = label_mapping[obj_name]
        if sem_label not in valid_semantic_mapping:
            sem_label = 39
        if (item['scene_id'], sem_label) not in sem_cache:
            sem_cache[(item['scene_id'], sem_label)] = 0
        sem_cache[(item['scene_id'], sem_label)] += 1
        obj_cache[(item["scene_id"], item["object_id"])] = True

    for item in scanrefer_json_data:
        scene_id = item['scene_id']
        obj_name = item["object_name"].replace("_", " ")
        sem_label = 39
        if obj_name in label_mapping:
            sem_label = label_mapping[obj_name]
        if sem_label not in valid_semantic_mapping:
            sem_label = 39
        assert sem_cache[(scene_id, sem_label)] >= 1
        item["eval_type"] = "unique" if sem_cache[(scene_id, sem_label)] == 1 else "multiple"
    # save the new json
    with open(file_path, "w") as f:
        json.dump(scanrefer_json_data, f, indent=2)


def load_scene(filename):
    d = torch.load(filename, weights_only=False)
    # return d['aabb_obj_ids'].tolist(), d['aabb_corner_xyz'].tolist()
    object_ids = d['aabb_obj_ids'].tolist()
    corner_xyz = d['aabb_corner_xyz'].tolist()

    ret = {}
    for i in range(len(object_ids)):
        object_id = str(object_ids[i])

        xs, ys, zs = zip(*corner_xyz[i])
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        z_min, z_max = min(zs), max(zs)

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2
        w = x_max - x_min
        h = y_max - y_min
        l = z_max - z_min

        ret[object_id] = (x_center, y_center, z_center, w, h, l)

    return ret


def embodiedscan_bbox_rescale(bbox, scale=1.0):
    center = np.array(bbox[:3]) * scale
    size = np.array(bbox[3:6]) * scale
    euler_zxy = np.array(bbox[6:9])

    bbox = center.tolist() + size.tolist() + euler_zxy.tolist()
    return bbox

def scanrefer_bbox_rescale(box, scale=1.0):
    center = np.array(box[:3]) * scale
    size = np.array(box[3:6]) * scale
    bbox = center.tolist() + size.tolist() + [0,0,0]
    return bbox

def match_scanrefer_bbox_with_embodiedscan(scan, scanrefer_bbox):
    instances = scan['instances']
    max_iou = 0
    instance_id = -1
    for i, instance in enumerate(instances):
        bbox1 = embodiedscan_bbox_rescale(instance["bbox_3d"], 100)
        bbox2 = scanrefer_bbox_rescale(scanrefer_bbox, 100)
        iou = EulerDepthInstance3DBoxes.overlaps(
            EulerDepthInstance3DBoxes(torch.tensor([bbox1]), convention="ZXY"),
            EulerDepthInstance3DBoxes(torch.tensor([bbox2]), convention="ZXY")
        )
        if max_iou < iou:
            max_iou = iou
            instance_id = i

    return instance_id, max_iou


def process_data_item(item, scan, desc, answer, images, box, split, object_json=None):
    image_tokens = "".join(["Frame-{}: <image>".format(i) for i in range(len(images))])
    # We select the frame with the clearest view of the object.
    prompt=f"""Localize the first clear frame in the video showing the object described in the text.
Text: {desc}
Output a JSON dictionary with the frame index in "frame" and its 3D bounding box in "bbox_3d" in the frame's coordinates.
"""
    question = f"{image_tokens}\n{prompt}"
    output = {
        "images": [image["img_path"] for image in images],
        "conversations": [
            {
                "value": question,
                "from": "human",
            },
            {
                "value": answer,
                "from": "gpt",
            },
        ],
        "metadata": {
            "dataset": "scanrefer",
            "question_type": item["eval_type"], 
            "ann_id": item["ann_id"],
            "object_id": item["object_id"],
        },
        "target": object_json if split == "train" else None,
        "gt_bbox": box,
        "prompt": prompt,
    }

    if args.split == "val" or args.include_cam_params:
        output.update({
            "cam2img": scan['cam2img'].tolist(),
            "cam2global": [x["cam2global"].tolist() for x in images],
            "axis_align_matrix": scan["axis_align_matrix"].tolist(), 
        })
    return output

@ray.remote
def main(data, args):
    id2scan = {}
    for split in ["train", "val", "test"]:
        with open(os.path.join(args.embodiedscan, f"embodiedscan_infos_{split}.pkl"), 'rb') as f:
            embodeidscan_data = pickle.load(f)
            # id2category = {v: k for k, v in embodeidscan_data['metainfo']['categories'].items()}
            id2scan.update({x["sample_idx"]: x for x in embodeidscan_data["data_list"]})

    split = args.split
    all_data = []
    scan2box = {}
    missing_num = 0
    for i, item in enumerate(tqdm(data)):
        if split == "test":
            box = None
        else:
            # load ground truth box
            scene_id = item['scene_id']
            if scene_id not in scan2box:
                scan2box[scene_id] = load_scene(os.path.join(args.scannet_dir, "pcd_with_object_aabbs", split, f"{scene_id}.pth"))
            box = scan2box[scene_id][item['object_id']]
            box = list(box) + [0, 0, 0]
        desc = item['description'].capitalize()
        
        scan = id2scan[f"scannet/{item['scene_id']}"]

        if split == "train":
            gt_instance_id, iou = match_scanrefer_bbox_with_embodiedscan(scan, box)
            if iou < 0.5:
                missing_num += 1
                print(f"Low iou: {item['scene_id']}, {item['object_id']}")
                continue
            
            for _ in range(2):
                images, frame_id = sample_images_and_best_view(scan, args.nframes, gt_instance_id)
                if frame_id == -1:
                    missing_num += 1
                    print(f"Missing instance: {item['scene_id']}, {item['object_id']}")
                    continue

                axis_align_matrix = np.array(scan['axis_align_matrix'])
                extrinsic = axis_align_matrix @ np.array(images[frame_id]["cam2global"])    # current camera to world
                bbox_3d_in_cam = _9dof_transform_world2cam(box, extrinsic, convention="ZXY")
                object_json = {
                    "frame": frame_id,
                    "bbox_3d": [round(x, 2) for i, x in enumerate(bbox_3d_in_cam)]
                }
                answer = f"```json\n\t{json.dumps(object_json)}\n```"
                output = process_data_item(item, scan, desc, answer, images, box, split=split, object_json=object_json)
                all_data.append(output)
        else:
            images = uniform_sample_images(scan['images'], args.nframes)
            answer = ""
            output = process_data_item(item, scan, desc, answer, images, box, split=split)
            all_data.append(output)

    return all_data, missing_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embodiedscan", type=str, default="/mnt/data0/zhengduo/data/embodiedscan-v2")
    parser.add_argument("--scanrefer_dir", type=str, default="/mnt/data0/zhengduo/data/scanrefer/")
    parser.add_argument("--scannet_dir", type=str, default="data/media/scannet/")
    parser.add_argument("--output_dir", type=str, default="data/new_train")
    parser.add_argument("--nframes", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--include_cam_params", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    args.label_mapping_file = os.path.join(args.scannet_dir, "metadata/scannetv2-labels.combined.tsv")
    args.valid_semantic_mapping = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]  # skip floor, wall and ceiling

    for split in ["train", "val"]:
        label_mapping = get_semantic_mapping_file(args.label_mapping_file, "nyu40")
        add_unique_multiple_labels_to_json(
            os.path.join(args.scanrefer_dir, f"ScanRefer_filtered_{split}.json"),
            label_mapping,
            args.valid_semantic_mapping,
        )

    random.seed(42)
    with open(os.path.join(args.scanrefer_dir, f"ScanRefer_filtered_{args.split}.json")) as f:
        data = json.load(f)

    ray.init()
    features = []
    for i in range(args.workers):
        features.append(main.remote(data[i::args.workers], args))
    
    all_data = []
    missing_num = 0
    for feature in tqdm(ray.get(features)):
        data, missing = feature
        all_data.extend(data)
        missing_num += missing
    
    with open(os.path.join(args.output_dir, f"scanrefer_{args.split}_{args.nframes}frames.json"), "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved to {os.path.join(args.output_dir, f'scanrefer_{args.split}_{args.nframes}frames.json')}")
    print(f"Missing num: {missing_num}")
    print(f"Total num: {len(all_data)}")
