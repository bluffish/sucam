import os
import subprocess

import cv2

import torch
import yaml
import pynvml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

from datasets.nuscenes import compile_data as compile_data_nuscenes

datasets = {
    'nuscenes': compile_data_nuscenes,
}


@torch.no_grad()
def run_loader(model, loader, config):
    predictions = []
    ground_truth = []

    with torch.no_grad():
        for images, segs, depths, intrinsics, extrinsics, labels in tqdm(loader, desc="Running validation"):
            if config['gt_depth']:
                b, c = depths.shape[:2]
                depths[depths == -1] = 0
                gt_depth = F.one_hot(depths, num_classes=112).permute(0, 1, 4, 2, 3)
                gt_depth = gt_depth.view(b * c, 112, 28, 60)
            else:
                gt_depth = None

            outs, seg_outs, depths = model.forward(images, intrinsics, extrinsics, gt_depth=gt_depth)
            outs = outs.detach().cpu()

            predictions.append(model.activate(outs))
            ground_truth.append(labels)

    return (torch.cat(predictions, dim=0),
            torch.cat(ground_truth, dim=0))


def save_pred(pred, label, out_path):
    cv2.imwrite(os.path.join(out_path, "pred.png"), pred[0, 0].detach().cpu().numpy() * 255)
    cv2.imwrite(os.path.join(out_path, "label.png"), label[0, 0].detach().cpu().numpy() * 255)


def apply_colormap_and_save(image_tensor, colormap, scale, output_file):
    # Normalize and apply colormap
    image_array = image_tensor.detach().cpu().numpy()
    normalized_image = (scale * image_array).astype(np.uint8)
    colored_image = colormap(normalized_image)
    # Convert from RGBA to RGB (if necessary)
    if colored_image.shape[-1] == 4:
        colored_image = colored_image[..., :3]
    # Convert to uint8
    final_image = (colored_image * 255).astype(np.uint8)
    # Save image using OpenCV
    cv2.imwrite(output_file, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))


def save_dep(depth, label, out_path):
    colormap = plt.cm.get_cmap('jet')
    scale = 255 / 40  # Assuming the depth values are to be scaled by this factor

    depth_image = depth[0].argmax(dim=0)
    label_image = label[0, 0]

    depth_file_path = os.path.join(out_path, "depth_pred.png")
    label_file_path = os.path.join(out_path, "depth_label.png")

    apply_colormap_and_save(depth_image, colormap, scale, depth_file_path)
    apply_colormap_and_save(label_image, colormap, scale, label_file_path)


def get_config(args):
    config = {}

    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    return config


def get_available_gpus(required_gpus=2):
    username = os.getlogin()
    available_gpus = []
    device_count = pynvml.nvmlDeviceGetCount()

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        compute_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)

        user_procs = [proc for proc in compute_procs if get_username_from_pid(proc.pid) == username]
        if not user_procs:
            available_gpus.append(i)

        if len(available_gpus) >= required_gpus:
            break

    return available_gpus[:required_gpus]


def get_username_from_pid(pid):
    try:
        proc = subprocess.run(['ps', '-o', 'user=', '-p', str(pid)], capture_output=True, text=True)
        return proc.stdout.strip()
    except subprocess.CalledProcessError:
        return None