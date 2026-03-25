import re
import os
import pandas as pd
from pathlib import Path
import yaml
import torch
from PIL import Image
from collections import defaultdict
from loguru import logger as eval_logger

import numpy as np
from typing import Union
from pytorch3d.ops import box3d_overlap
from pytorch3d.transforms import euler_angles_to_matrix
from terminaltables import AsciiTable

cate8 = [
    "chair", "cabinet", "table", "bin", "couch", "bed", "bathtub", "toilet",
]

cate20 = [
    "chair", "pillow", "cabinet", "table", "lamp", "couch", "desk", "stand", "bed", "backpack",
    "bathtub", "ottoman", "dresser", "bin", "toilet", "refrigerator", "stove", "microwave", "monitor", "computer",
]
cate31 = [
    "chair", "pillow", "cabinet", "table", "lamp", "couch", "desk", "stand", "bed", "backpack",
    "bathtub", "ottoman", "dresser", "bin", "toilet", "refrigerator", "stove", "microwave", "monitor", "computer",
    "window", "shelf", "curtain", "plant", "stairs", "picture", "book", "bottle", "lamp", "towl", "sink",
]

def rotation_3d_in_euler(points, angles, convention, return_mat=False, clockwise=False):
    """Rotate points by angles according to axis.

    Args:
        points (np.ndarray | torch.Tensor | list | tuple ):
            Points of shape (N, M, 3).
        angles (np.ndarray | torch.Tensor | list | tuple):
            Vector of angles in shape (N, 3)
        return_mat: Whether or not return the rotation matrix (transposed).
            Defaults to False.
        clockwise: Whether the rotation is clockwise. Defaults to False.

    Raises:
        ValueError: when the axis is not in range [0, 1, 2], it will
            raise value error.

    Returns:
        (torch.Tensor | np.ndarray): Rotated points in shape (N, M, 3).
    """
    batch_free = len(points.shape) == 2
    if batch_free:
        points = points[None]

    if len(angles.shape) == 1:
        angles = angles.expand(points.shape[:1] + (3, ))
        # angles = torch.full(points.shape[:1], angles)

    assert len(points.shape) == 3 and len(angles.shape) == 2 \
        and points.shape[0] == angles.shape[0], f'Incorrect shape of points ' \
        f'angles: {points.shape}, {angles.shape}'

    assert points.shape[-1] in [2, 3], \
        f'Points size should be 2 or 3 instead of {points.shape[-1]}'

    rot_mat_T = euler_angles_to_matrix(angles, convention)  # N, 3,3
    # rot_mat_T = torch.tensor(R.from_euler(convention, angles.cpu().numpy()).as_matrix(), dtype=torch.float32)
    rot_mat_T = rot_mat_T.transpose(-2, -1)

    if clockwise:
        raise NotImplementedError('clockwise')

    if points.shape[0] == 0:
        points_new = points
    else:
        points_new = torch.bmm(points, rot_mat_T)

    if batch_free:
        points_new = points_new.squeeze(0)

    if return_mat:
        if batch_free:
            rot_mat_T = rot_mat_T.squeeze(0)
        return points_new, rot_mat_T
    else:
        return points_new


class EulerDepthInstance3DBoxes:
    """3D boxes of instances in Depth coordinates.

    We keep the "Depth" coordinate system definition in MMDet3D just for
    clarification of the points coordinates and the flipping augmentation.

    Coordinates in Depth:

    .. code-block:: none

                    up z    y front (alpha=0.5*pi)
                       ^   ^
                       |  /
                       | /
                       0 ------> x right (alpha=0)

    The relative coordinate of bottom center in a Depth box is (0.5, 0.5, 0),
    and the yaw is around the z axis, thus the rotation axis=2.
    The yaw is 0 at the positive direction of x axis, and decreases from
    the positive direction of x to the positive direction of y.
    Also note that rotation of DepthInstance3DBoxes is counterclockwise,
    which is reverse to the definition of the yaw angle (clockwise).

    Attributes:
        tensor (torch.Tensor): Float matrix of N x box_dim.
        box_dim (int): Integer indicates the dimension of a box
            Each row is (x, y, z, x_size, y_size, z_size, alpha, beta, gamma).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    def __init__(self,
                 tensor,
                 convention="ZXY",
                 box_dim=9,
                 with_yaw=True,
                 origin=(0.5, 0.5, 0.5)):

        if isinstance(tensor, torch.Tensor):
            device = tensor.device
        else:
            device = torch.device('cpu')
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that
            # does not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((0, box_dim)).to(dtype=torch.float32,
                                                     device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == box_dim, tensor.size()

        if tensor.shape[-1] == 6:
            # If the dimension of boxes is 6, we expand box_dim by padding
            # (0, 0, 0) as a fake euler angle.
            assert box_dim == 6
            fake_rot = tensor.new_zeros(tensor.shape[0], 3)
            tensor = torch.cat((tensor, fake_rot), dim=-1)
            self.box_dim = box_dim + 3
        elif tensor.shape[-1] == 7:
            assert box_dim == 7
            fake_euler = tensor.new_zeros(tensor.shape[0], 2)
            tensor = torch.cat((tensor, fake_euler), dim=-1)
            self.box_dim = box_dim + 2
        else:
            assert tensor.shape[-1] == 9
            self.box_dim = box_dim
        self.tensor = tensor.clone()

        self.origin = origin
        if origin != (0.5, 0.5, 0.5):
            dst = self.tensor.new_tensor((0.5, 0.5, 0.5))
            src = self.tensor.new_tensor(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)
        self.with_yaw = with_yaw
        self.convention = convention

    def __len__(self) -> int:
        """int: Number of boxes in the current object."""
        return self.tensor.shape[0]

    def __getitem__(self, item: Union[int, slice, np.ndarray, torch.Tensor]):
        """
        Args:
            item (int or slice or np.ndarray or Tensor): Index of boxes.

        Note:
            The following usage are allowed:

            1. `new_boxes = boxes[3]`: Return a `Boxes` that contains only one
               box.
            2. `new_boxes = boxes[2:10]`: Return a slice of boxes.
            3. `new_boxes = boxes[vector]`: Where vector is a
               torch.BoolTensor with `length = len(boxes)`. Nonzero elements in
               the vector will be selected.

            Note that the returned Boxes might share storage with this Boxes,
            subject to PyTorch's indexing semantics.

        Returns:
            :obj:`BaseInstance3DBoxes`: A new object of
            :class:`BaseInstance3DBoxes` after indexing.
        """
        original_type = type(self)
        if isinstance(item, int):
            return original_type(self.tensor[item].view(1, -1),
                                 box_dim=self.box_dim,
                                 with_yaw=self.with_yaw)
        b = self.tensor[item]
        assert b.dim() == 2, \
            f'Indexing on Boxes with {item} failed to return a matrix!'
        return original_type(b, box_dim=self.box_dim, with_yaw=self.with_yaw)

    @property
    def dims(self) -> torch.Tensor:
        """Tensor: Size dimensions of each box in shape (N, 3)."""
        return self.tensor[:, 3:6]

    @classmethod
    def overlaps(cls, boxes1, boxes2, mode='iou', eps=1e-4):
        """Calculate 3D overlaps of two boxes.

        Note:
            This function calculates the overlaps between ``boxes1`` and
            ``boxes2``, ``boxes1`` and ``boxes2`` should be in the same type.

        Args:
            boxes1 (:obj:`EulerInstance3DBoxes`): Boxes 1 contain N boxes.
            boxes2 (:obj:`EulerInstance3DBoxes`): Boxes 2 contain M boxes.
            mode (str): Mode of iou calculation. Defaults to 'iou'.
            eps (bool): Epsilon. Defaults to 1e-4.

        Returns:
            torch.Tensor: Calculated 3D overlaps of the boxes.
        """
        assert isinstance(boxes1, EulerDepthInstance3DBoxes)
        assert isinstance(boxes2, EulerDepthInstance3DBoxes)
        assert type(boxes1) == type(boxes2), '"boxes1" and "boxes2" should' \
            f'be in the same type, got {type(boxes1)} and {type(boxes2)}.'

        assert mode in ['iou']

        rows = len(boxes1)
        cols = len(boxes2)
        if rows * cols == 0:
            return boxes1.tensor.new(rows, cols)

        corners1 = boxes1.corners
        corners2 = boxes2.corners
        _, iou3d = box3d_overlap(corners1, corners2, eps=eps)
        return iou3d

    @property
    def corners(self):
        """torch.Tensor: Coordinates of corners of all the boxes
        in shape (N, 8, 3).

        Convert the boxes to corners in clockwise order, in form of
        ``(x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)``

        .. code-block:: none

                                           up z
                            front y           ^
                                 /            |
                                /             |
                  (x0, y1, z1) + -----------  + (x1, y1, z1)
                              /|            / |
                             / |           /  |
               (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                            |  /      .   |  /
                            | / origin    | /
               (x0, y0, z0) + ----------- + --------> right x
                                          (x1, y0, z0)
        """
        if self.tensor.numel() == 0:
            return torch.empty([0, 8, 3], device=self.tensor.device)

        dims = self.dims
        corners_norm = torch.from_numpy(
            np.stack(np.unravel_index(np.arange(8), [2] * 3),
                     axis=1)).to(device=dims.device, dtype=dims.dtype)

        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        # use relative origin
        assert self.origin == (0.5, 0.5, 0.5), \
            'self.origin != (0.5, 0.5, 0.5) needs to be checked!'
        corners_norm = corners_norm - dims.new_tensor(self.origin)
        corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

        # rotate
        corners = rotation_3d_in_euler(corners, self.tensor[:, 6:], self.convention)

        corners += self.tensor[:, :3].view(-1, 1, 3)
        return corners


with open(Path(__file__).parent / "default_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
media_dir = yaml.safe_load("".join(safe_data))["metadata"]["media_dir"]
# embodiedscan_path = yaml.safe_load("".join(safe_data))["metadata"]["embodiedscan_path"]
# with open(embodiedscan_path, "rb") as f:
#     data = pickle.load(f)["data_list"]
#     id2scene = {sample["sample_id"]: sample for sample in data}

def threedod_doc_to_visual(doc):
    image_files = doc["images"]
    images = [
        Image.open(
            os.path.join(media_dir, image_file)
        ).convert("RGB")
        for image_file in image_files
    ]
    return [images]


def threedod_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    prompt = doc["conversations"][0]["value"].replace("<image>", "")
    return prompt



def compute_ap(gt_bbox_dict, pred_bbox_dict, iou_threshold=0.25):

    all_tp = defaultdict()
    all_fp = defaultdict()
    all_fn = defaultdict()
    used_gt = defaultdict(set)
    for category in pred_bbox_dict:
        for bbox in pred_bbox_dict[category]:
            gt_box_match = -1
            max_iou = 0
            for i, gt_box in enumerate(gt_bbox_dict[category]):
                if i in used_gt[category]:
                    continue
                try:
                    iou = EulerDepthInstance3DBoxes.overlaps(
                        EulerDepthInstance3DBoxes(torch.tensor([bbox]), convention="ZXY"),
                        EulerDepthInstance3DBoxes(torch.tensor([gt_box]), convention="ZXY")
                    )
                except Exception as e:
                    eval_logger.error(f"Error calculating IOU: {e}")
                    iou = 0
                if iou > max_iou:
                    max_iou = iou
                    gt_box_match = i
            
            if max_iou > iou_threshold:
                used_gt[category].add(gt_box_match)
                all_tp[category] = all_tp.get(category, 0) + 1
            else:
                all_fp[category] = all_fp.get(category, 0) + 1
    
    for category in gt_bbox_dict:
        for i, gt_box in enumerate(gt_bbox_dict[category]):
            if i not in used_gt[category]:
                all_fn[category] = all_fn.get(category, 0) + 1        

    categories = set(pred_bbox_dict.keys()) | set(gt_bbox_dict.keys())
    ret = {
        category: {
            "tp": all_tp.get(category, 0),
            "fp": all_fp.get(category, 0),
            "fn": all_fn.get(category, 0),
        }
        for category in categories
    }
    return ret
   

def threedod_process_results(doc, results):

    lines = results[0].strip('\n').strip("```").strip("json").strip("\n").split("\n")
    pred = []
    for line in lines:
        line = line.strip().strip(",")
        if "bbox_3d" not in line and "label" not in line:
            continue
        try:
            pred_box = eval(line)
            pred.append(pred_box)
        except Exception as e:
            eval_logger.error(f"Error parsing prediction bbox: {line}, Error: {e}")

    pred_bbox_dict = defaultdict(list)
    gt_bbox_dict = defaultdict(list)

    for bbox in doc["boxes"]:
        gt_bbox_dict[bbox["label"]].append(bbox["bbox_3d"])
    
    for bbox in pred:
        try:
            pred_bbox = np.array(bbox["bbox_3d"], dtype=float)
            # pred_bbox[:6] = pred_bbox[:6] / 100.
            pred_bbox_dict[bbox["label"]].append(pred_bbox)
        except Exception as e:
            eval_logger.error(f"Error parsing prediction bbox: {bbox}, Error: {e}")

    ret = compute_ap(gt_bbox_dict, pred_bbox_dict)

    return {
        "threedod_score": {
            "result": ret,
            "gt_labels": list(set([bbox["label"] for bbox in doc["boxes"]])),
            "images": doc["images"],
        }
    }

def threedod_aggregate_results(results):

    all_results = defaultdict(list)
    for result in results:
        for k, v in result["result"].items():
            all_results[k].append(v)
    eval_logger.info("AP Results:")


    all_metrics = {}
    table_data = [
        ["Category", "Precision", "Recall", "F1 Score"]  # 表头
    ]

    for k in cate31:
        v = all_results.get(k, [])
        tp = sum([result["tp"] for result in v])
        fp = sum([result["fp"] for result in v])
        fn = sum([result["fn"] for result in v])        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        all_metrics[f"{k}_precision"] = precision
        all_metrics[f"{k}_recall"] = recall
        all_metrics[f"{k}_f1"] = f1
        table_data.append([
            k,
            f"{precision:.4f}",
            f"{recall:.4f}",
            f"{f1:.4f}"
        ])
    table = AsciiTable(table_data)
    table.title = "Metrics Per Category"
    print(table.table)
    

    avg_table_data = [
        ["Split", "Avg Precision", "Avg Recall", "Avg F1"]
    ]
    for split_name, split_categories in zip(['cate8', 'cate20', 'cate31'], [cate8, cate20, cate31]):
        precisions = [all_metrics.get(f"{k}_precision", 0) for k in split_categories]
        recalls = [all_metrics.get(f"{k}_recall", 0) for k in split_categories]
        f1s = [all_metrics.get(f"{k}_f1", 0) for k in split_categories]
    
        avg_table_data.append([
            split_name,
            f"{np.mean(precisions):.4f}",
            f"{np.mean(recalls):.4f}",
            f"{np.mean(f1s):.4f}"
        ])

    table = AsciiTable(avg_table_data)
    print(table.table)

    return np.mean([all_metrics.get(f"{k}_f1", 0) for k in cate31])