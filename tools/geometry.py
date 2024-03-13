import cv2
import numpy as np
import torch
from nuscenes.utils.data_classes import Box
from PIL import Image
from pyquaternion import Quaternion
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from scipy.ndimage import distance_transform_edt


def dist_true(masks):
    batch_size = masks.size(0)
    distances_batch = []

    for i in range(batch_size):
        mask_np = masks[i].cpu().numpy()

        if mask_np.sum() == 0:
            distances_batch.append(torch.full((200, 200), 199))
        else:
            inverted_mask = np.logical_not(mask_np)
            distances = distance_transform_edt(inverted_mask)
            distances_batch.append(torch.from_numpy(distances))

    return torch.stack(distances_batch, dim=0)


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


def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)

    return min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)


def fill_convex_hull(image, points, fill_value=1.0):

    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    pts = np.array(hull_points, 'int32')
    pts = pts.reshape((-1, 1, 2))

    cv2.fillPoly(image, [pts], fill_value)


def get_image_points(locations, intrinsic, extrinsic, image_size=(224, 480)):
    num_points = locations.shape[0]
    points = np.hstack((locations, np.ones((num_points, 1))))

    points_camera = np.dot(points, extrinsic.T)
    points_camera = points_camera[:, [1, 2, 0]]
    points_camera[:, 1] *= -1

    points_img = np.dot(points_camera, intrinsic.T)

    points_img[:, 0] /= points_img[:, 2] + 1e-8
    points_img[:, 1] /= points_img[:, 2] + 1e-8

    mask = (points_img[:, 0] >= 0) & (points_img[:, 0] <= 480) & \
           (points_img[:, 1] >= 0) & (points_img[:, 1] <= 224) & \
           (points_camera[:, 2] <= 0)

    return points_img[:, 0:2], mask


def adjust_carla(extrinsic):
    adjust_roll = Rotation.from_euler('x', [-90], degrees=True)
    adjust_yaw = Rotation.from_euler('z', [-90], degrees=True)
    e = Rotation.from_matrix(extrinsic[:3, :3])

    extrinsic[:3, :3] = (adjust_roll * e * adjust_yaw).as_matrix()


def adjust_nuscenes(extrinsic):
    adjust_roll = Rotation.from_euler('x', [90], degrees=True)
    adjust_yaw = Rotation.from_euler('z', [90], degrees=True)
    e = Rotation.from_matrix(extrinsic[:3, :3])

    extrinsic[:3, :3] = (adjust_roll * e * adjust_yaw).as_matrix()


def draw_cam(
        translation, rotation, size,
        intrinsic, extrinsic,
        image_size=(224, 480),
        dataset='carla'
):
    extrinsic = extrinsic.copy()
    cam = np.zeros(image_size)
    box = Box(translation, size, Quaternion(rotation))
    corners = box.corners().T

    if dataset == 'carla':
        adjust_carla(extrinsic)
    else:
        adjust_nuscenes(extrinsic)

    corners, mask = get_image_points(corners, intrinsic, extrinsic, image_size=image_size)

    if len(corners[mask]) > 1:
        fill_convex_hull(cam, corners)

    return cam


def draw_bev(
        translation, rotation, size,
        bev_resolution=(0.5, 0.5, 20.),
        bev_start_position=(-49.75, -49.75, 0.),
        bev_size=(200, 200),
):
    bev_resolution = np.array(bev_resolution)
    bev_start_position = np.array(bev_start_position)
    bev = np.zeros(bev_size)
    box = Box(translation, size, Quaternion(rotation))

    pts = box.bottom_corners()[:2].T
    pts = np.round((pts - bev_start_position[:2] + bev_resolution[:2] / 2.0)
                   / bev_resolution[:2]).astype(np.int32)

    pts[:, [1, 0]] = pts[:, [0, 1]]
    cv2.fillPoly(bev, [pts], 1.0)

    return bev


def find_bounding_boxes(mask):
    num_labels, labels = cv2.connectedComponents(mask.astype('uint8'))

    bounding_boxes = []

    for label in range(1, num_labels):
        group_mask = np.where(labels == label, 255, 0).astype('uint8')
        contours, _ = cv2.findContours(group_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            bounding_box = cv2.minAreaRect(contour)

            # fixes problem of void regions in BEV camera

            (center_x, center_y), (width, height), angle = bounding_box
            area = width * height

            if area > 50:
                continue

            bounding_boxes.append(((center_x, center_y), (width+2, height+2), angle))

    return bounding_boxes


def draw_bounding_boxes(bounding_boxes, dim=(200, 200)):
    mask = np.zeros(dim)

    for bb in bounding_boxes:
        bb = cv2.boxPoints(bb).astype('int')
        cv2.fillPoly(mask, [bb], 1.0)

    return mask
