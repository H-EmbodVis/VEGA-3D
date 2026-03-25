import random
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from scipy.spatial.transform import Rotation as R


def get_frame_id(img_path):
    """
    Currently only support ScanNet image path format.
    """
    try:
        ret = img_path.split("/")[-1].split(".")[0]
        return int(ret)
    except Exception as e:
        import pdb; pdb.set_trace()


def embodiedscan_bbox_to_o3d_geo(box):
    """
    Convert an EmbodiedScan 9-DOF bounding box to an Open3D OrientedBoundingBox.

    Args:
        box (list or np.ndarray): A 9-element sequence [x, y, z, dx, dy, dz, rx, ry, rz]
                                  representing center, size, and rotation (Euler ZXY).

    Returns:
        open3d.geometry.OrientedBoundingBox: The corresponding Open3D geometry object.
    """
    if isinstance(box, list):
        box = np.array(box)
    center = box[:3].reshape(3, 1)
    scale = box[3:6].reshape(3, 1)
    rot = box[6:].reshape(3, 1)
    rot_mat = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_zxy(
        rot)
    geo = o3d.geometry.OrientedBoundingBox(center, rot_mat, scale)

    return geo


def o3d_geo_to_9dof(obb, convention):
    """
    Convert an Open3D OrientedBoundingBox to a 9-DOF bounding box list.

    Args:
        obb (open3d.geometry.OrientedBoundingBox): The Open3D geometry object.
        convention (str): The Euler angle convention (e.g., 'ZXY').

    Returns:
        list: A 9-element list [x, y, z, dx, dy, dz, rx, ry, rz] representing
              center, size, and rotation.
    """
    center = obb.center
    size = obb.extent
    rotation = R.from_matrix(obb.R)
    euler = rotation.as_euler(convention)
    ret = list(center) + list(size) + list(euler)
    return ret


def _9dof_transform_world2cam(box, extrinsic, convention):
    """
    Transform a 9-DOF bounding box from world coordinates to camera coordinates.

    Args:
        box (list or np.ndarray): A 9-element sequence [x, y, z, dx, dy, dz, rx, ry, rz]
                                  representing the box in world coordinates.
        extrinsic (np.ndarray): The 4x4 camera-to-world transformation matrix.
        convention (str): The Euler angle convention (e.g., 'ZXY').

    Returns:
        list: A 9-element list representing the box in camera coordinates.
    """
    center = box[:3]
    extent = box[3:6]
    euler = box[6:]

    global2cam = np.linalg.inv(extrinsic)
    new_center = (global2cam @ np.array(list(center) + [1]).reshape(4, 1)).reshape(4)[:3].tolist()
    new_rot = global2cam[:3, :3] @ R.from_euler(convention, euler).as_matrix()
    new_euler = R.from_matrix(new_rot).as_euler(convention).tolist()
    new_bbox = new_center + extent + new_euler
    return new_bbox


def _inside_box(box, point):
    """Check if any points are in the box.

    Args:
        box (open3d.geometry.OrientedBoundingBox): Oriented Box.
        point (np.ndarray): N points represented by nx4 array (x, y, z, 1).

    Returns:
        bool: The result.
    """
    point_vec = o3d.utility.Vector3dVector(point[:, :3])
    inside_idx = box.get_point_indices_within_bounding_box(point_vec)
    if len(inside_idx) > 0:
        return True
    return False

def calculate_box_projection_area(box, extrinsic, intrinsic):
    camera_pos_in_world = (
        extrinsic @ np.array([0, 0, 0, 1]).reshape(4, 1)).transpose()
    if _inside_box(box, camera_pos_in_world):
        return 0

    corners = np.asarray(box.get_box_points())
    corners = corners[[0, 1, 7, 2, 3, 6, 4, 5]]
    corners = np.concatenate(
        [corners, np.ones((corners.shape[0], 1))], axis=1)
    
    extrinsic_w2c = np.linalg.inv(extrinsic)
    corners_cam = extrinsic_w2c @ corners.transpose()
    corners_img = intrinsic @ corners_cam
    corners_img = corners_img.transpose()
    corners_cam = corners_cam.transpose()
    corners_pixel = np.zeros((corners_img.shape[0], 2))
    for i in range(corners_img.shape[0]):
        corners_pixel[i] = corners_img[i][:2] / np.abs(corners_img[i][2])

    mask = corners_cam[:, 2] > 0
    valid_points = corners_pixel[mask]
    image_width = intrinsic[0, 2] * 2
    image_height = intrinsic[1, 2] * 2
    # Clip the points to be within the image boundaries
    valid_points[:, 0] = np.clip(valid_points[:, 0], 0, image_width-1)
    valid_points[:, 1] = np.clip(valid_points[:, 1], 0, image_height-1)
    
    if len(valid_points) < 3 or \
        np.all((valid_points[:, 0] == 0) | (valid_points[:, 0] == image_width-1)) or \
        np.all((valid_points[:, 1] == 0) | (valid_points[:, 1] == image_height-1)):
        return 0.0
    
    hull = ConvexHull(valid_points)
    polygon = Polygon(valid_points[hull.vertices])
    return polygon.area


def uniform_sample_images(images, nframe):
    """
    Uniformly sample nframe images from the list of images.
    """
    idx = random.randint(0, len(images))
    images = images[idx:] + images[:idx]
    if len(images) < nframe:
        return images
    interval = len(images) // nframe
    return images[::interval][:nframe]



def sample_images_and_best_view(scan, nframes, gt_instance_id):
    """
    Sample nframes images containing the target instance and identify the best view.
    """
    new_images = uniform_sample_images(scan["images"], nframes)
    flg = False
    for _ in range(20):
        if any([gt_instance_id in image["visible_instance_ids"] for image in new_images]):
            flg = True
            break
        new_images = uniform_sample_images(scan["images"], nframes)
    if not flg:
        new_images = new_images[:-1]
        for img in scan["images"]:
            if gt_instance_id in img["visible_instance_ids"]:
                new_images.append(img)
                flg = True
                break
        new_images = sorted(new_images, key=lambda x: x["img_path"])
    
    frame_id = -1 
    max_area = 0
    obb = embodiedscan_bbox_to_o3d_geo(scan["instances"][gt_instance_id]["bbox_3d"])

    for i, img in enumerate(new_images):
        extrinsic = np.array(scan["axis_align_matrix"]) @ np.array(img["cam2global"])
        intrinsic = np.array(scan["cam2img"])
        if gt_instance_id in img["visible_instance_ids"]:
            try:
                area = calculate_box_projection_area(
                    obb,
                    extrinsic,
                    intrinsic,
                )
                if area > max_area:
                    max_area = area
                    frame_id = i
            except Exception as e:
                print(e)
                pass

    return new_images, frame_id 