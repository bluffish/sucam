import argparse
import random
from time import time, sleep

from tensorboardX import SummaryWriter

from models.model import Model
from tools.metrics import *
from tools.utils import *
from datasets.nuscenes import *
from time import time, sleep
from tqdm import tqdm


if __name__ == "__main__":
    ndt = compile_data(True, "trainval", "../data/nuscenes", "vehicle", batch_size=1024, num_workers=64, drop_last=False)

    ndt.dataset.gen_labels = True
    t = time()
    for _ in tqdm(ndt.dataset):
        pass
    print(time()-t)

    ndv = compile_data(False, "trainval", "../data/nuscenes", "vehicle", batch_size=1024, num_workers=64, drop_last=False)

    ndv.dataset.gen_labels = True
    t = time()
    for _ in tqdm(ndv.dataset):
        pass
    print(time()-t)