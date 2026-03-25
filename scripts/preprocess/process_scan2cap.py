"""
Refer to https://github.com/embodied-generalist/embodied-generalist/blob/main/data/datasets.py

scanrefer
- ScanRefer_filtered_train.json
- ScanRefer_filtered_val.json

scannet
- mask
- pcd_with_global_alignment
- vg

"""

import os
import json
import torch
import pickle
import random
import argparse
import numpy as np
from tqdm import tqdm
from scipy import sparse
from collections import defaultdict
from utils import uniform_sample_images


def convert_pc_to_box(obj_pc):
    # converting point clouds into 6 DoF bounding box
    xmin = np.min(obj_pc[:,0])
    ymin = np.min(obj_pc[:,1])
    zmin = np.min(obj_pc[:,2])
    xmax = np.max(obj_pc[:,0])
    ymax = np.max(obj_pc[:,1])
    zmax = np.max(obj_pc[:,2])
    center = [(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2]
    box_size = [xmax-xmin, ymax-ymin, zmax-zmin]
    return center, box_size

def get_3d_box_corners(center, box_size):
    ''' box_size is array(l,w,h), heading_angle is radius clockwise from pos x axis, center is xyz of box center
        output (8,3) array for 3D box cornders
        Similar to utils/compute_orientation_3d
    '''
    l,w,h = box_size
    # x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    # y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    # z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    z_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    corners_3d = np.vstack([x_corners,y_corners,z_corners])
    corners_3d[0,:] = corners_3d[0,:] + center[0]
    corners_3d[1,:] = corners_3d[1,:] + center[1]
    corners_3d[2,:] = corners_3d[2,:] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d

def get_box3d_min_max(corner):
    ''' Compute min and max coordinates for 3D bounding box
        Note: only for axis-aligned bounding boxes

    Input:
        corners: numpy array (8,3), assume up direction is Z (batch of N samples)
    Output:
        box_min_max: an array for min and max coordinates of 3D bounding box IoU

    '''

    min_coord = corner.min(axis=0)
    max_coord = corner.max(axis=0)
    x_min, x_max = min_coord[0], max_coord[0]
    y_min, y_max = min_coord[1], max_coord[1]
    z_min, z_max = min_coord[2], max_coord[2]
    
    return x_min, x_max, y_min, y_max, z_min, z_max


def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is Z
        corners2: numpy array (8,3), assume up direction is Z
    Output:
        iou: 3D bounding box IoU

    '''

    x_min_1, x_max_1, y_min_1, y_max_1, z_min_1, z_max_1 = get_box3d_min_max(corners1)
    x_min_2, x_max_2, y_min_2, y_max_2, z_min_2, z_max_2 = get_box3d_min_max(corners2)
    xA = np.maximum(x_min_1, x_min_2)
    yA = np.maximum(y_min_1, y_min_2)
    zA = np.maximum(z_min_1, z_min_2)
    xB = np.minimum(x_max_1, x_max_2)
    yB = np.minimum(y_max_1, y_max_2)
    zB = np.minimum(z_max_1, z_max_2)
    inter_vol = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0) * np.maximum((zB - zA), 0)
    box_vol_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1) * (z_max_1 - z_min_1)
    box_vol_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2) * (z_max_2 - z_min_2)
    iou = inter_vol / (box_vol_1 + box_vol_2 - inter_vol + 1e-8)

    return iou


def load_masks(scannet_base, scan_id, pcds):
    mask_path = os.path.join(scannet_base, 'mask', f'{str(scan_id)}.mask.npz')
    obj_masks = np.array(sparse.load_npz(mask_path).todense())[:50, :]
    obj_pcds_pred = []
    for i in range(obj_masks.shape[0]):
        mask = obj_masks[i]
        obj_pcds_pred.append(pcds[mask == 1, :].astype(float))

    return obj_pcds_pred


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

        ret[object_id] = [x_center, y_center, z_center, w, h, l]

    return ret



def main(args):
    id2scan = {}
    for split in ["train", "val", "test"]:
        with open(os.path.join(args.embodiedscan, f"embodiedscan_infos_{split}.pkl"), 'rb') as f:
            data = pickle.load(f)
            id2category = {v: k for k, v in data['metainfo']['categories'].items()}
            id2scan.update({x["sample_idx"]: x for x in data["data_list"]})

    for split in ["train", "val"]:
        n_miss = 0
        with open(os.path.join(args.scanrefer_dir, f"ScanRefer_filtered_{split}.json")) as f:
            data = json.load(f)
        
        # preprocess annotations for val split. for each instance, there are many annotations
        if split == "val":
            instance_annotations = defaultdict(list)
            for item in data:
                key = f"{item['scene_id']}|{item['object_id']}|{item['object_name']}"
                instance_annotations[key].append(item['description'])

        scan2box = {}
        scan2pred = {}
        all_data = []
        visible_instances = set()
        for i, item in enumerate(tqdm(data)):
            scene_id = item['scene_id']

            # skip duplicate
            key = f"{item['scene_id']}|{item['object_id']}|{item['object_name']}"
            if split != 'train' and key in visible_instances:
                continue
            visible_instances.add(key)

            # load ground truth box
            if scene_id not in scan2box:
                scan2box[scene_id] = load_scene(os.path.join(args.scannet_dir, "pcd_with_object_aabbs", split, f"{item['scene_id']}.pth"))

            gt_box = scan2box[scene_id][item['object_id']]

            # load predicted boxes
            if split == "val":   
                if scene_id not in scan2pred:
                    pcd_data = torch.load(os.path.join(args.scannet_dir,
                                    'pcd_with_object_aabbs', split, f'{scene_id}.pth'), weights_only=False)
                    points, colors = pcd_data["xyz"], pcd_data["rgb"]
                    colors = colors / 127.5 - 1
                    pcds = np.concatenate([points, colors], 1)    

                    pred_pcds = load_masks(args.scannet_dir, scene_id, pcds)
                    scan2pred[scene_id] = [convert_pc_to_box(pcd) for pcd in pred_pcds]

                boxes = scan2pred[scene_id]

                pred_box = None
                max_iou = -1
                for center, sz in boxes:
                    iou = box3d_iou(
                        get_3d_box_corners(center, sz),
                        get_3d_box_corners(gt_box[:3], gt_box[3:])
                    )

                    if iou > max_iou:
                        max_iou = iou
                        pred_box = center + sz
                
                if max_iou < args.threshold:
                    print(f"{key} is missing")
                    n_miss += 1

            input_box = gt_box if split == "train" else pred_box
            scan = id2scan[f"scannet/{item['scene_id']}"]
            images = uniform_sample_images(scan['images'], args.nframes)
            axis_align_matrix = np.array(scan['axis_align_matrix'])
            reference_frame = images[-1] if args.reference_frame == "last" else images[0]
            extrinsic = axis_align_matrix @ np.array(reference_frame["cam2global"])
            global2cam = np.linalg.inv(extrinsic)
            tranformed_coord = (global2cam @ np.array(input_box[:3] + [1]).reshape(4, 1)).reshape(4)[:3].tolist()
            desc = item['description'].capitalize()
            transformed_coord = [round(i, 2) for i in tranformed_coord]

            new_item = {
                "conversations": [
                    {
                        "value": f"{''.join(['<image>'] * len(images))}\nCarefully watch the video and describe the object located at {transformed_coord} in detail.",
                        "from": "human",
                    },
                    {
                        "value": f"{desc}",
                        "from": "gpt",
                    },
                ],
                "images": [img['img_path'] for img in images],
                "input_box": input_box,
                "gt_box": gt_box,
                "iou": max_iou if split == "val" else 1,
                "metadata": {
                    "dataset": "scan2cap",
                    "question_type": item["eval_type"], 
                    "ann_id": item["ann_id"],
                    "object_id": item["object_id"],
                }
            }

            if split == "val":
                new_item['annotations'] = instance_annotations[key]
                new_item.update({
                    "cam2img": scan['cam2img'].tolist(),
                    "cam2global": [x["cam2global"].tolist() for x in images],
                    "axis_align_matrix": scan["axis_align_matrix"].tolist(), 
                })
            all_data.append(new_item)
        
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, f'scan2cap_{split}_{args.nframes}frames.json'), 'w') as f:
            json.dump(all_data, f)
        print(f"total {len(all_data)} items.")
        print(f"total {n_miss} miss.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scanrefer_dir", type=str, default="/mnt/data0/zhengduo/data/scanrefer/")
    parser.add_argument("--scannet_dir", type=str, default="/mnt/data0/zhengduo/data/scannet")
    parser.add_argument("--embodiedscan", type=str, default="/mnt/data0/zhengduo/data/embodiedscan-v2")
    parser.add_argument("--reference_frame", type=str, default="first")
    parser.add_argument("--output_dir", type=str, default="data/train")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--nframes", type=int, default=32)
    args = parser.parse_args()

    random.seed(42)
    main(args)