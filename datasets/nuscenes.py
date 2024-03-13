import os
import random
import warnings

import torch
import torchvision

from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from nuscenes.utils.data_classes import Box, LidarPointCloud
from functools import reduce
from nuscenes.utils.data_io import load_bin_file

from shapely.errors import ShapelyDeprecationWarning

from tools.geometry import *

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


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
    return (pts[2] > 0) &\
        (pts[0] > 1) & (pts[0] < W - 1) &\
        (pts[1] > 1) & (pts[1] < H - 1)


def get_lidar_data(nusc, sample_rec, nsweeps, min_distance):
    """
    Returns at most nsweeps of lidar in the ego frame.
    Returned tensor is 5(x, y, z, reflectance, dt) x N
    Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
    """
    points = np.zeros((5, 0))

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data']['LIDAR_TOP']
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                        inverse=True)

    # Aggregate current and previous sweeps.
    sample_data_token = sample_rec['data']['LIDAR_TOP']
    current_sd_rec = nusc.get('sample_data', sample_data_token)
    for _ in range(nsweeps):
        # Load up the pointcloud and remove points close to the sensor.
        current_pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, current_sd_rec['filename']))
        points_label = np.fromfile(os.path.join(nusc.dataroot, current_sd_rec['filename']), dtype=np.uint8)

        current_pc.remove_close(min_distance)

        # Get past pose.
        current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                            Quaternion(current_pose_rec['rotation']), inverse=False)

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
        current_pc.transform(trans_matrix)

        # Add time vector which can be used as a temporal feature.
        time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
        times = time_lag * np.ones((1, current_pc.nbr_points()))

        new_points = np.concatenate((current_pc.points, times), 0)
        points = np.concatenate((points, new_points), 1)

        # Abort if there are no previous sweeps.
        if current_sd_rec['prev'] == '':
            break
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    return points

class NuScenesDataset(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, pos_class, ind=False, ood=False, pseudo=False, yaw=180):
        self.ind = ind
        self.ood = ood
        self.pseudo = pseudo
        self.pos_class = pos_class
        self.yaw = yaw

        self.pseudo_ood = ["vehicle.bicycle", "static_object.bicycle_rack"]
        self.true_ood = ["vehicle.motorcycle"]

        self.all_ood = self.true_ood + self.pseudo_ood

        self.nusc = nusc
        self.is_train = is_train

        self.dataroot = self.nusc.dataroot

        self.mode = 'train' if self.is_train else 'val'

        self.scenes = self.get_scenes()
        self.ixes = self.prepro()

        self.maps = {map_name: NuScenesMap(dataroot=self.dataroot, map_name=map_name) for map_name in [
            "singapore-hollandvillage",
            "singapore-queenstown",
            "boston-seaport",
            "singapore-onenorth",
        ]}

        self.augmentation_parameters = self.get_resizing_and_cropping_parameters()

        self.to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            [-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5], [-10.0, 10.0, 20.0]
        )

        self.bev_resolution, self.bev_start_position, self.bev_dimension = (
            bev_resolution, bev_start_position, bev_dimension
        )

        self.cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

        self.sm = {}
        for rec in nusc.scene:
            log = nusc.get('log', rec['log_token'])
            self.sm[rec['name']] = log['location']

    def get_scenes(self):
        split = {'v1.0-trainval': {True: 'train', False: 'val'},
                 'v1.0-mini': {True: 'mini_train', False: 'mini_val'}, }[
            self.nusc.version
        ][self.is_train]

        scenes = create_splits_scenes()[split]

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]
        samples = [samp for samp in samples if self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        records = []

        for rec in samples:
            ego_pose = self.nusc.get('ego_pose', rec['data']['LIDAR_TOP'])

            ego_coord = ego_pose['translation']

            is_true_ood = False
            is_pseudo_ood = False

            for tok in rec['anns']:
                inst = self.nusc.get('sample_annotation', tok)

                box_coord = inst['translation']

                if max(abs(ego_coord[0] - box_coord[0]), abs(ego_coord[1] - box_coord[1])) > 50 or int(
                        inst['visibility_token']) <= 2:
                    continue

                if inst['category_name'] in self.true_ood:
                    is_true_ood = True

                if inst['category_name'] in self.pseudo_ood:
                    is_pseudo_ood = True

            if self.ind and not is_pseudo_ood and not is_true_ood:
                records.append(rec)
            if self.ood and is_true_ood:
                records.append(rec)
            if self.pseudo and is_pseudo_ood:
                records.append(rec)

        return records

    @staticmethod
    def get_resizing_and_cropping_parameters():
        original_height, original_width = 900, 1600
        final_height, final_width = 224, 480

        resize_scale = .3
        resize_dims = (int(original_width * resize_scale), int(original_height * resize_scale))
        resized_width, resized_height = resize_dims

        crop_h = 46
        crop_w = int(max(0, (resized_width - final_width) / 2))
        crop = (crop_w, crop_h, crop_w + final_width, crop_h + final_height)

        return {'scale_width': resize_scale,
                'scale_height': resize_scale,
                'resize_dims': resize_dims,
                'crop': crop}

    def get_lidar_data(self, rec, nsweeps):
        pts = get_lidar_data(self.nusc, rec,
                             nsweeps=nsweeps, min_distance=2.2)
        return torch.Tensor(pts)[:3]

    def get_input_data(self, rec):
        images = []
        intrinsics = []
        extrinsics = []
        depths = []
        segs = []

        points = self.get_lidar_data(rec, nsweeps=5)

        # panoptic_file = os.path.join(self.nusc.dataroot,
        #                                     self.nusc.get('panoptic', rec['data']['LIDAR_TOP'])['filename'])
        # points_label = np.fromfile(panoptic_file, dtype=np.uint8)

        for cam in self.cameras:
            camera_sample = self.nusc.get('sample_data', rec['data'][cam])
            sensor_sample = self.nusc.get('calibrated_sensor', camera_sample['calibrated_sensor_token'])

            intrinsic = torch.Tensor(sensor_sample['camera_intrinsic'])

            q = sensor_sample['rotation']

            adjust_yaw = Rotation.from_euler('z', [self.yaw], degrees=True)
            sensor_rotation = Rotation.from_quat([q[1], q[2], q[3], q[0]]).inv() * adjust_yaw

            sensor_translation = np.array(sensor_sample['translation'])

            extrinsic = np.eye(4, dtype=np.float32)
            extrinsic[:3, :3] = sensor_rotation.as_matrix()
            extrinsic[:3, 3] = sensor_translation
            extrinsic = np.linalg.inv(extrinsic)

            image = Image.open(os.path.join(self.dataroot, camera_sample['filename']))

            image = resize_and_crop_image(image, resize_dims=self.augmentation_parameters['resize_dims'],
                                          crop=self.augmentation_parameters['crop'])
            normalized_image = self.to_tensor(image)

            top_crop = self.augmentation_parameters['crop'][1]
            left_crop = self.augmentation_parameters['crop'][0]

            intrinsic = update_intrinsics(
                intrinsic, top_crop, left_crop,
                scale_width=self.augmentation_parameters['scale_width'],
                scale_height=self.augmentation_parameters['scale_height']
            )

            cam_points = ego_to_cam(points,
                                 torch.Tensor(Quaternion(sensor_sample['rotation']).rotation_matrix),
                                 torch.Tensor(sensor_sample['translation']), intrinsic)
            mask = get_only_in_img_mask(cam_points, 224, 480)
            cam_points = cam_points[:, mask]

            w, h = 30, 14
            depth = np.zeros((h, w))
            depth.fill(40)

            for i in range(cam_points.shape[1]):
                yco = int(cam_points[0, i] // (224 // h))
                xco = int(cam_points[1, i] // (480 // w))

                depth[xco, yco] = min(cam_points[2, i], depth[xco][yco])

            depths.append(torch.tensor(depth))
            images.append(normalized_image)
            intrinsics.append(intrinsic)
            extrinsics.append(torch.tensor(extrinsic))

        return (torch.stack(images, dim=0),
                torch.ones((1, 1)),
                torch.stack(depths, dim=0),
                torch.stack(intrinsics, dim=0),
                torch.stack(extrinsics, dim=0))

    def _get_top_lidar_pose(self, rec):
        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        yaw = Quaternion(egopose['rotation']).yaw_pitch_roll[0]
        rot = Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse
        return trans, rot

    def get_label(self, rec):
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])

        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse

        vehicles = np.zeros(self.bev_dimension[:2])
        ood = np.zeros(self.bev_dimension[:2])

        for token in rec['anns']:
            inst = self.nusc.get('sample_annotation', token)

            if int(inst['visibility_token']) == 1:
                continue

            if inst['category_name'] in self.all_ood:
                pts, _ = self.get_region(inst, trans, rot)
                cv2.fillPoly(ood, [pts], 1.0)
            elif 'vehicle' in inst['category_name']:
                pts, _ = self.get_region(inst, trans, rot)
                cv2.fillPoly(vehicles, [pts], 1.0)

        road, lane = self.get_map(rec)
        empty = np.ones(self.bev_dimension[:2])

        if self.pos_class == 'vehicle':
            empty[vehicles == 1] = 0
            label = np.stack((vehicles, empty))
        elif self.pos_class == 'road':
            empty[road == 1] = 0
            label = np.stack((road, empty))
        elif self.pos_class == 'lane':
            empty[lane == 1] = 0
            label = np.stack((lane, empty))
        elif self.pos_class == 'all':
            empty[vehicles == 1] = 0
            empty[lane == 1] = 0
            empty[road == 1] = 0

            road[vehicles == 1] = 0
            road[lane == 1] = 0
            lane[vehicles == 1] = 0
            label = np.stack((vehicles, road, lane, empty))

        return torch.tensor(label.copy()), torch.tensor(ood[None])

    def get_region(self, instance_annotation, ego_translation, ego_rotation):
        box = Box(instance_annotation['translation'], instance_annotation['size'],
                  Quaternion(instance_annotation['rotation']))

        box.translate(ego_translation)
        box.rotate(ego_rotation)

        pts = box.bottom_corners()[:2].T
        pts = np.round(
            (pts - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0) / self.bev_resolution[:2]).astype(
            np.int32)
        pts[:, [1, 0]] = pts[:, [0, 1]]

        z = box.bottom_corners()[2, 0]

        return pts, z

    def get_map(self, rec):
        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        map_name = self.sm[self.nusc.get('scene', rec['scene_token'])['name']]
        center = np.array([egopose['translation'][0], egopose['translation'][1]])

        rota = quaternion_yaw(Quaternion(egopose['rotation'])) / np.pi * 180
        road = np.any(self.maps[map_name].get_map_mask((center[0], center[1], 100, 100), rota, ['road_segment', 'lane'],
                                                       canvas_size=self.bev_dimension[:2]), axis=0).T

        lane = np.any(
            self.maps[map_name].get_map_mask((center[0], center[1], 100, 100), rota, ['road_divider', 'lane_divider'],
                                             canvas_size=self.bev_dimension[:2]), axis=0).T

        return road.astype(np.uint8), lane.astype(np.uint8)

    def __len__(self):
        return len(self.ixes)

    def __getitem__(self, index):
        rec = self.ixes[index]
        images, segs, depths, intrinsics, extrinsics = self.get_input_data(rec)
        labels, oods = self.get_label(rec)

        return images, segs, depths, intrinsics, extrinsics, labels, oods


def get_nusc(version, dataroot):
    dataroot = os.path.join(dataroot, version)
    nusc = NuScenes(version=f'v1.0-{version}', dataroot=dataroot, verbose=False)

    return nusc, dataroot


def compile_data(set, version, dataroot, pos_class, batch_size=8, num_workers=16, seed=0, is_train=False, yaw=180):
    if set == "train":
        ind, ood, pseudo, is_train = True, False, False, True
    elif set == "val":
        ind, ood, pseudo, is_train = True, False, False, False
    elif set == "train_aug":
        ind, ood, pseudo, is_train = False, False, True, True
    elif set == "val_aug":
        ind, ood, pseudo, is_train = False, False, True, False
    elif set == "train_comb":
        ind, ood, pseudo, is_train = True, False, True, True
    elif set == "val_comb":
        ind, ood, pseudo, is_train = True, False, True, False
    elif set == "train_full":
        ind, ood, pseudo, is_train = True, True, True, True
    elif set == "val_full":
        ind, ood, pseudo, is_train = True, True, True, False
    elif set == "ood":
        ind, ood, pseudo, is_train = False, True, False, False
    elif set == "test":
        ind, ood, pseudo, is_train = True, False, False, False
    elif set == "ood_test":
        ind, ood, pseudo, is_train = True, True, False, False
    else:
        raise NotImplementedError(f"Dataset {set} not exist.")

    nusc, dataroot = get_nusc("trainval", dataroot)

    data = NuScenesDataset(nusc, is_train, pos_class, ind=ind, ood=ood, pseudo=pseudo, yaw=yaw)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if version == 'mini':
        g = torch.Generator()
        g.manual_seed(seed)

        sampler = torch.utils.data.RandomSampler(data, num_samples=256, generator=g)

        loader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            sampler=sampler,
            pin_memory=True,
        )
    else:
        loader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )

    return loader