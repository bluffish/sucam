import cv2
import numpy as np
import torch
from nuscenes.utils.data_classes import Box
from PIL import Image
from pyquaternion import Quaternion
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from scipy.ndimage import distance_transform_edt


def euler_to_quaternion(yaw, pitch, roll):
    yaw, pitch, roll = np.radians(yaw),  np.radians(pitch),  np.radians(roll)
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qx, qy, qz, qw]


def resize_and_crop_image(img, resize_dims, crop):
    img = img.resize(resize_dims, resample=Image.BILINEAR)
    img = img.crop(crop)
    return img


def mask(img, target):
    m = np.all(img == target, axis=2).astype(int)
    return m


def update_intrinsics(intrinsics, top_crop=0.0, left_crop=0.0, scale_width=1.0, scale_height=1.0):
    updated_intrinsics = intrinsics.clone()

    updated_intrinsics[0, 0] *= scale_width
    updated_intrinsics[0, 2] *= scale_width
    updated_intrinsics[1, 1] *= scale_height
    updated_intrinsics[1, 2] *= scale_height

    updated_intrinsics[0, 2] -= left_crop
    updated_intrinsics[1, 2] -= top_crop

    return updated_intrinsics


def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    bev_resolution = np.array([row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = np.array([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = np.array([(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]], dtype=np.int32)

    return bev_resolution, bev_start_position, bev_dimension


def warp_features(x, flow, mode='nearest', spatial_extent=None):
    if flow is None:
        return x
    b, c, h, w = x.shape
    angle = flow[:, 5].clone()
    translation = flow[:, :2].clone()

    translation[:, 0] /= spatial_extent[0]
    translation[:, 1] /= spatial_extent[1]

    translation[:, 0] *= -1

    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)

    transformation = torch.stack([cos_theta, -sin_theta, translation[:, 1],
                                  sin_theta, cos_theta, translation[:, 0]], dim=-1).view(b, 2, 3)

    grid = torch.nn.functional.affine_grid(transformation, size=x.shape, align_corners=False)
    warped_x = torch.nn.functional.grid_sample(x, grid.float(), mode=mode, padding_mode='zeros', align_corners=False)

    return warped_x


def ego_to_cam(points, rot, trans, intrins):
    """Transform points (3 x N) from ego frame into a pinhole camera
    """
    points = points - trans.unsqueeze(1)
    points = rot.permute(1, 0).matmul(points)

    points = intrins.matmul(points)
    points[:2] /= points[2:3]

    return points


def get_only_in_img_mask(pts, H, W):
    """pts should be 3 x N
    """
    return (pts[2] > 0) & \
        (pts[0] > 1) & (pts[0] < W - 1) & \
        (pts[1] > 1) & (pts[1] < H - 1)