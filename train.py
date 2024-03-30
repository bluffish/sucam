import argparse
import random
from time import time, sleep

from tensorboardX import SummaryWriter

from models.model import Model
from tools.metrics import *
from tools.utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.multiprocessing.set_sharing_strategy('file_system')

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def train():
    train_loader = datasets[config['dataset']](
        True, split, dataroot, config['pos_class'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        seed=config['seed'],
    )

    val_loader = datasets[config['dataset']](
        False, split, dataroot, config['pos_class'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        seed=config['seed'],
    )

    model = Model(
        config['gpus'],
        backbone=config['backbone'],
        loss_type=config['loss'],
        use_dep=config['dep'],
        use_seg=config['seg'],
    )

    model.opt = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    if 'pretrained' in config:
        model.load(torch.load(config['pretrained']))
        print(f"Loaded pretrained weights: {config['pretrained']}")

    if not config['no_scheduler']:
        print("Using Scheduler")

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            model.opt,
            div_factor=10,
            pct_start=.3,
            final_div_factor=10,
            max_lr=config['learning_rate'],
            epochs=config['num_epochs'],
            steps_per_epoch=len(train_loader.dataset) // config['batch_size']
        )
    else:
        scheduler = None

    print("--------------------------------------------------")
    print(f"Using GPUS: {config['gpus']}")
    print(f"Train loader: {len(train_loader.dataset)}")
    print(f"Val loader: {len(val_loader.dataset)}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Output directory: {config['logdir']} ")
    print(f"Using loss {config['loss']}")
    print("--------------------------------------------------")

    writer = SummaryWriter(logdir=config['logdir'])

    if config['dep']:
        os.makedirs(os.path.join(config['logdir'], "dep"), exist_ok=True)
    if config['seg']:
        os.makedirs(os.path.join(config['logdir'], "seg"), exist_ok=True)

    writer.add_text("config", str(config))

    step = 0

    for epoch in range(config['num_epochs']):
        model.train()
        t_0_ep = time()

        writer.add_scalar('train/epoch', epoch, step)

        for images, segs, depths, intrinsics, extrinsics, labels in train_loader:

            t_0 = time()

            outs, seg_outs, dep_outs, loss, seg_loss, dep_loss = model.train_step(images, segs, depths, intrinsics, extrinsics, labels, use_gt_depth=config['gt_depth'])
            step += 1

            if scheduler is not None:
                scheduler.step()

            if step % 10 == 0:
                print(f"[{epoch}]", step, loss.item(), time()-t_0)
                writer.add_scalar('train/step_time', time() - t_0, step)
                writer.add_scalar('train/loss', loss, step)
                writer.add_scalar('train/dep_loss', dep_loss, step)
                writer.add_scalar('train/seg_loss', seg_loss, step)

                save_pred(model.activate(outs), labels, config['logdir'])

                if config['dep']:
                    save_dep(dep_outs, depths, os.path.join(config['logdir'], "dep"))
                if config['seg']:
                    save_pred(seg_outs.sigmoid().unsqueeze(0), segs, os.path.join(config['logdir'], "seg"))

            if step % 50 == 0:
                _, _, iou = get_iou(model.activate(outs).cpu(), labels)
                rse = get_rse(dep_outs.argmax(dim=1).flatten().cpu().float(), depths.flatten().cpu().float())

                print(f"[{epoch}] {step}", "IOU: ", iou, "RSE: ", rse.item())
                writer.add_scalar(f'train/iou', iou, step)
                writer.add_scalar(f'train/depth_rse', rse, step)

        print(f"Epoch Time: {time()-t_0_ep}")

        if config['no_val']:
            continue

        model.eval()

        preds, labels = run_loader(model, val_loader, config)

        _, _, iou = get_iou(preds, labels)
        writer.add_scalar(f'val/iou', iou, epoch)
        print(f"Validation mIOU: {iou}")

        model.save(os.path.join(config['logdir'], f'{epoch}.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset")
    parser.add_argument("backbone")
    parser.add_argument('-q', '--queue', default=False, action='store_true')
    parser.add_argument('-g', '--gpus', nargs='+', default=[7], type=int)
    parser.add_argument('-l', '--logdir', default='test', type=str)
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('-s', '--split', default="trainval", required=False, type=str)

    parser.add_argument('-p', '--pretrained', required=False, type=str)
    parser.add_argument('-e', '--num_epochs', default=20, type=int)
    parser.add_argument('-c', '--pos_class', default="vehicle", required=False, type=str)
    parser.add_argument('--seg', default=False, action='store_true')
    parser.add_argument('--dep', default=False, action='store_true')
    parser.add_argument('--no_val', default=False, action='store_true')
    parser.add_argument('--gt_depth', default=False, action='store_true')

    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--learning_rate', default=4e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-7, type=float)
    parser.add_argument('--no_scheduler', default=False, action='store_true')

    parser.add_argument('--seed', default=0, required=False, type=int)
    parser.add_argument('--loss', default="focal", required=False, type=str)

    args = parser.parse_args()

    config = get_config(args)

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    config['mixed'] = False

    if config['queue']:
        pynvml.nvmlInit()
        print("Waiting for suitable GPUs...")

        required_gpus = 2
        while True:
            available_gpus = get_available_gpus(required_gpus=required_gpus)
            if len(available_gpus) == required_gpus:
                print(f"Running program on GPUs {available_gpus}...")
                config['gpus'] = available_gpus
                break
            else:
                sleep(random.randint(30, 90))

        pynvml.nvmlShutdown()

    if config['backbone'] == 'cvt':
        torch.backends.cudnn.enabled = False

    for gpu in config['gpus']:
        torch.inverse(torch.ones((1, 1), device=gpu))

    split = args.split
    dataroot = f"../data/{config['dataset']}"

    train()
