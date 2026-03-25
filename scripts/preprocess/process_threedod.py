import os
import json
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import numpy as np
import open3d as o3d
import random
from utils import embodiedscan_bbox_to_o3d_geo, get_frame_id, o3d_geo_to_9dof


def create_question_answer(n_images, instances, args):
    if args.reference_frame == "first":
        instruction = f"Detect the 3D bounding boxes in the camera coordinate system of the first frame."
    elif args.reference_frame == "last":
        instruction = f"Detect the 3D bounding boxes in the camera coordinate system of the last frame."    
    else:
        raise NotImplementedError(f"Reference frame {args.reference_frame} not supported.")
    question = "".join(["<image>\n"] * n_images) + f"""{instruction}
Output a json list where each entry contains the object name in "label" and its 3D bounding box in "bbox_3d".
The 3D bounding box format should be [x_center, y_center, z_center, x_size, y_size, z_size, yaw, roll, pitch]."""


    def format_instance(instance):
        box_3d = []
        for i, x in enumerate(instance["bbox_3d_in_cam"]):
            x = round(x, 2)
            box_3d.append(x)

        item = {
            "label": instance["label"],
            "bbox_3d": box_3d,
        }
        return json.dumps(item)

    instances_str = ",\n\t".join([format_instance(instance) for instance in instances])
    answer = "```json\n[\n\t" + instances_str + "\n]```"
    return question, answer


def process_data_item(select_frames, sample, id2category, args):
    select_instances = []

    for frame in select_frames:
        for instance_id in frame["visible_instance_ids"]:
            if instance_id not in select_instances:
                select_instances.append(instance_id)
        
    select_instances = [
        {
            "instance_id": instance_id,
            **sample['instances'][instance_id]
        } for instance_id in list(select_instances)
    ]
    select_instances = [instance for instance in select_instances if id2category[instance["bbox_label_3d"]] not in ["wall", "ceiling", "floor", "object"]]
    if args.reference_frame == "first":
        reference_frame = select_frames[0]
    else:
        reference_frame = select_frames[-1]
    axis_align_matrix = np.array(sample["axis_align_matrix"])
    extrinsic = axis_align_matrix @ np.array(reference_frame["cam2global"])
    intrinsic = np.array(sample['cam2img'])
    global2cam = np.linalg.inv(extrinsic)

    new_instances = []
    for instance in select_instances:
        geo = embodiedscan_bbox_to_o3d_geo(instance["bbox_3d"])
        R_in_cam = global2cam[:3,:3] @ geo.R
        center_in_cam = global2cam @ np.array([*geo.center, 1]).reshape(4, 1)
        geo_in_cam = o3d.geometry.OrientedBoundingBox(center_in_cam[:3].reshape(3), R_in_cam, geo.extent)
        bbox_3d_in_cam = o3d_geo_to_9dof(geo_in_cam, convention="ZXY")
        new_instances.append({
            "label": id2category[instance["bbox_label_3d"]],
            "bbox_3d": instance["bbox_3d"],
            "bbox_3d_in_cam": bbox_3d_in_cam,
        })

    images = [frame["img_path"] for frame in select_frames]
    cam2global = [frame["cam2global"] for frame in select_frames]
    question, answer = create_question_answer(len(images), new_instances, args)
    output = {
        "conversations": [
            {
                "from": "human",
                "value": question,
            },
            {
                "from": "gpt",
                "value": answer,                        
            }
        ],
        "images": images,
        "boxes": [
            {
                "bbox_3d": item["bbox_3d_in_cam"],
                "label": item["label"],
            }
            for item in new_instances
        ]
    }
    if args.split == "val" or args.include_cam_params:
        output.update({
            "cam2img": sample['cam2img'].tolist(),
            "cam2global": [x.tolist() for x in cam2global],
            "axis_align_matrix": sample["axis_align_matrix"].tolist(), 
        })
    return output

def main(args):
    with open(os.path.join(args.embodiedscan, f"embodiedscan_infos_{args.split}.pkl"), 'rb') as f:
        data = pickle.load(f)

    id2category = {v:k for k, v in data['metainfo']['categories'].items()}

    all_data = []
    nums = []
    for sample in tqdm(data["data_list"]):
        if not sample['sample_idx'].startswith("scannet"):
            continue
        # ensure the frames is consecutive
        def get_consecutive_frames(all_frames):
            last_frame_idx = -10
            ret = []
            for frame in all_frames:
                frame_idx = get_frame_id(frame["img_path"])
                if frame_idx == last_frame_idx + 10:
                    ret.append(frame)
                else:
                    yield ret
                    ret = [frame]
                last_frame_idx = frame_idx
            yield ret
        
        items = []
        for frames in get_consecutive_frames(sample["images"]):
            for i in range(0, len(frames)+1):
                # n_sample = random.randint(args.nframe, args.max_frames)
                n_sample = args.nframe
                if args.reference_frame == "first":
                    select_frames = frames[i: i+n_sample*args.base_interval: args.base_interval]
                elif args.reference_frame == "last":
                    select_frames = frames[i-n_sample*args.base_interval :i: args.base_interval]
                if len(select_frames) < n_sample:
                    continue
                item = process_data_item(select_frames, sample, id2category, args)
                items.append(item)

        # random sample 10 samples form items
        if args.split == "val":
            if len(items) > 10:
                items = random.sample(items, 10)
        nums.append(len(items))
        all_data.extend(items)
    print("Average number of items: ", sum(nums) / len(nums))
    print(f"Scan num: {len(nums)}")
    all_categories = []
    for item in all_data:
        all_categories.extend([bbox["label"] for bbox in item["boxes"]])
    
    from collections import Counter
    counter = Counter(all_categories)
    # from the counter, get the top 20 categories
    top_20 = counter.most_common(20)
    print("Top 20 categories: ", top_20)

    print(f"Data size: {len(all_data)}")
    with open(os.path.join(args.output_dir, f"scannet_det_{args.split}_{args.nframe}frames.json"), 'w') as f:
        json.dump(all_data, f, indent=2)
    print(f'Output to {os.path.join(args.output_dir, f"scannet_det_{args.split}_{args.nframe}frames.json")}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embodiedscan", type=str, default="/mnt/data0/zhengduo/data/embodiedscan-v2")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--nframe", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="data/new_train")
    parser.add_argument("--base_interval", type=int, default=3)
    parser.add_argument("--reference_frame", type=str, default="first", choices=["last", "first"])
    parser.add_argument("--include_cam_params", action='store_true', help="Whether to include camera parameters in the output json.")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    main(args)