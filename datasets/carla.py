import json
import math
import os
import random

import torchvision
from torch.utils.data import Subset

from tools.geometry import *


class CarlaDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, is_train, pos_class):
        self.is_train = is_train
        self.return_info = False
        self.pos_class = pos_class

        self.data_path = data_path

        self.mode = 'train' if self.is_train else 'val'

        self.vehicles = len(os.listdir(os.path.join(self.data_path, 'agents')))
        self.ticks = len(os.listdir(os.path.join(self.data_path, 'agents/0/back_camera')))
        self.offset = 0

        self.to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            [-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5], [-10.0, 10.0, 20.0]
        )

        self.bev_resolution, self.bev_start_position, self.bev_dimension = (
            bev_resolution, bev_start_position, bev_dimension
        )

    def get_input_data(self, index, agent_path):
        images = []
        segs = []
        depths = []
        intrinsics = []
        extrinsics = []

        with open(os.path.join(agent_path, 'sensors.json'), 'r') as f:
            sensors = json.load(f)

        for sensor_name, sensor_info in sensors['sensors'].items():
            if sensor_info["sensor_type"] == "sensor.camera.rgb" and sensor_name != "birds_view_camera":
                image = Image.open(os.path.join(agent_path + sensor_name, f'{index}.png'))
                depth_i = Image.open(os.path.join(agent_path + sensor_name[:-7] + "_depth", f'{index}.png'))
                seg_i = Image.open(os.path.join(agent_path + sensor_name[:-7] + "_semantic", f'{index}.png'))

                intrinsic = torch.tensor(sensor_info["intrinsic"])
                translation = np.array(sensor_info["transform"]["location"])
                rotation = sensor_info["transform"]["rotation"]

                rotation[0] += 90
                rotation[2] -= 90

                r = Rotation.from_euler('zyx', rotation, degrees=True)

                extrinsic = np.eye(4, dtype=np.float32)
                extrinsic[:3, :3] = r.as_matrix()
                extrinsic[:3, 3] = translation
                extrinsic = np.linalg.inv(extrinsic)

                normalized_image = self.to_tensor(image)

                seg = np.array(seg_i)
                empty = np.ones(seg.shape[:2])
                vehicles = mask(seg, (0, 0, 142))
                road = mask(seg, (128, 64, 128))
                lane = mask(seg, (157, 234, 50))

                if np.sum(vehicles) < 5:
                    lane |= mask(seg, (50, 234, 157))
                    vehicles |= mask(seg, (142, 0, 0))

                depth = np.array(depth_i)
                depth = depth[:, :, 0] + depth[:, :, 1] * 256. + depth[:, :, 2] * 256. * 256.
                depth /= 256. * 256. * 256. - 1.
                depth *= 1000

                images.append(normalized_image)
                segs.append(torch.tensor(np.stack((vehicles, road, lane, empty))))
                depths.append(torch.tensor(depth))
                intrinsics.append(intrinsic)
                extrinsics.append(torch.tensor(extrinsic))
                image.close()

        return (torch.stack(images, dim=0),
                torch.stack(segs, dim=0),
                torch.stack(depths, dim=0),
                torch.stack(intrinsics, dim=0),
                torch.stack(extrinsics, dim=0))

    def get_label(self, index, agent_path):
        label_r = Image.open(os.path.join(agent_path + "bev_semantic", f'{index}.png'))
        label = np.array(label_r)
        label_r.close()

        empty = np.ones(self.bev_dimension[:2])

        road = mask(label, (128, 64, 128))
        lane = mask(label, (157, 234, 50))
        vehicles = mask(label, (0, 0, 142))

        if np.sum(vehicles) < 5:
            lane = mask(label, (50, 234, 157))
            vehicles = mask(label, (142, 0, 0))

        ood = mask(label, (0, 0, 0))
        bounding_boxes = find_bounding_boxes(ood)
        ood = draw_bounding_boxes(bounding_boxes)

        if self.pos_class == 'vehicle':
            empty[vehicles == 1] = 0
            label = np.stack((vehicles, empty))

        elif self.pos_class == 'road':
            road[lane == 1] = 1
            road[vehicles == 1] = 1

            # this is a refinement step to remove some impurities in the label caused by small objects
            road = (road * 255).astype(np.uint8)
            kernel_size = 2

            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            road = cv2.dilate(road, kernel, iterations=1)
            road = cv2.erode(road, kernel, iterations=1)
            empty[road == 1] = 0

            label = np.stack((road, empty))
        elif self.pos_class == 'lane':
            empty[lane == 1] = 0

            label = np.stack((lane, empty))
        elif self.pos_class == 'all':
            empty[vehicles == 1] = 0
            empty[lane == 1] = 0
            empty[road == 1] = 0
            label = np.stack((vehicles, road, lane, empty))

        return label, ood[None]

    def __len__(self):
        return self.ticks * self.vehicles

    def __getitem__(self, index):
        agent_number = math.floor(index / self.ticks)
        agent_path = os.path.join(self.data_path, f"agents/{agent_number}/")
        index = (index + self.offset) % self.ticks

        images, segs, depths, intrinsics, extrinsics = self.get_input_data(index, agent_path)
        labels, ood = self.get_label(index, agent_path)

        if self.return_info:
            return images, intrinsics, extrinsics, labels, ood, {
                'agent_number': agent_number,
                'agent_path': agent_path,
                'index': index
            }

        return images, segs, depths, intrinsics, extrinsics, labels, ood


def compile_data(set, version, dataroot, pos_class, batch_size=8, num_workers=16, is_train=False, seed=0, yaw=-1):
    data = CarlaDataset(os.path.join(dataroot, set), is_train, pos_class)
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
            pin_memory=False,
        )

    return loader