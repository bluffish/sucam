import os
import subprocess

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import pynvml
from tqdm import tqdm

from datasets.carla import compile_data as compile_data_carla
from datasets.nuscenes import compile_data as compile_data_nuscenes

from models.baseline import Baseline
from models.evidential import Evidential
from models.ensemble import Ensemble
from models.dropout import Dropout
from models.postnet import Postnet


colors = torch.tensor([
    [0, 0, 255],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 0],
])

models = {
    'baseline': Baseline,
    'evidential': Evidential,
    'ensemble': Ensemble,
    'dropout': Dropout,
    'postnet': Postnet,
}

datasets = {
    'nuscenes': compile_data_nuscenes,
    'carla': compile_data_carla,
}

n_classes, classes = 2, ["vehicle", "background"]
weights = torch.tensor([2, 1])


def change_params(config):
    global classes, n_classes, weights

    if config['pos_class'] == 'vehicle':
        n_classes, classes = 2, ["vehicle", "background"]
        weights = torch.tensor([2., 1.])
    elif config['pos_class'] == 'road':
        n_classes, classes = 2, ["road", "background"]
        weights = torch.tensor([1., 1.])
    elif config['pos_class'] == 'lane':
        n_classes, classes = 2, ["lane", "background"]
        weights = torch.tensor([5., 1.])
    elif config['pos_class'] == 'all':
        n_classes, classes = 4, ["vehicle", "road", "lane", "background"]
        weights = torch.tensor([3., 1., 2., 1.])
    else:
        raise NotImplementedError("Invalid Positive Class")

    return classes, n_classes, weights


@torch.no_grad()
def run_loader(model, loader):
    predictions = []
    seg_preds = []
    ground_truth = []
    seg_labels = []
    oods = []
    aleatoric = []
    epistemic = []
    raw = []

    with torch.no_grad():
        for images, segs, depths, intrinsics, extrinsics, labels, ood in tqdm(loader, desc="Running validation"):

            outs, seg_outs, depths = model.forward(images, intrinsics, extrinsics)
            outs = outs.detach().cpu()

            if model.use_seg:
                seg_outs = seg_outs.detach().cpu()
                seg_preds.append(model.activate(seg_outs))
            else:
                seg_preds.append(torch.zeros((1, 1, 1)))

            predictions.append(model.activate(outs))
            ground_truth.append(labels)
            seg_labels.append(segs)
            oods.append(ood)
            aleatoric.append(model.aleatoric(outs))
            epistemic.append(model.epistemic(outs))
            raw.append(outs)

    return (torch.cat(predictions, dim=0),
            torch.cat(seg_preds, dim=0),
            torch.cat(ground_truth, dim=0),
            torch.cat(seg_labels, dim=0),
            torch.cat(oods, dim=0),
            torch.cat(aleatoric, dim=0),
            torch.cat(epistemic, dim=0),
            torch.cat(raw, dim=0))


def map_rgb(onehot, ego=False):
    dense = onehot.permute(1, 2, 0).detach().cpu().numpy().argmax(-1)

    rgb = np.zeros((*dense.shape, 3))
    for label, color in enumerate(colors):
        rgb[dense == label] = color

    if ego:
        rgb[94:106, 98:102] = (0, 255, 255)

    return rgb


def save_unc(u_score, u_true, out_path, score_name, true_name):
    u_score = u_score.detach().cpu().numpy()
    u_true = u_true.numpy()

    cv2.imwrite(
        os.path.join(out_path, true_name),
        u_true[0, 0] * 255
    )

    cv2.imwrite(
        os.path.join(out_path, score_name),
        cv2.cvtColor((plt.cm.inferno(u_score[0, 0]) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    )


def save_pred(pred, label, out_path, ego=False):
    if pred.shape[1] != 2:
        pred = map_rgb(pred[0], ego=ego)
        label = map_rgb(label[0], ego=ego)
        cv2.imwrite(os.path.join(out_path, "pred.png"), pred)
        cv2.imwrite(os.path.join(out_path, "label.png"), label)

        return pred, label
    else:
        cv2.imwrite(os.path.join(out_path, "pred.png"), pred[0, 0].detach().cpu().numpy() * 255)
        cv2.imwrite(os.path.join(out_path, "label.png"), label[0, 0].detach().cpu().numpy() * 255)


def get_mis(pred, label):
    return (pred.argmax(dim=1) != label.argmax(dim=1)).unsqueeze(1)


def get_config(args):
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

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