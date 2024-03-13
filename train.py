import argparse
import random
from time import time, sleep

from tensorboardX import SummaryWriter
from tools.metrics import *
from tools.utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.multiprocessing.set_sharing_strategy('file_system')

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

print(torch.__version__)

def train():
    classes, n_classes, weights = change_params(config)

    if config['loss'] == 'focal':
        config['learning_rate'] *= 4

    if config['ood']:
        train_set = "train_comb"
        val_set = "val_comb"
    else:
        train_set = "train"
        val_set = "val"

    if 'train_set' in config:
        train_set = config['train_set']
    if 'val_set' in config:
        val_set = config['val_set']

    if config['backbone'] == 'lss':
        yaw = 0
    elif config['backbone'] == 'cvt':
        yaw = 180

    train_loader = datasets[config['dataset']](
        train_set, split, dataroot, config['pos_class'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        is_train=True,
        seed=config['seed'],
        yaw=yaw
    )

    val_loader = datasets[config['dataset']](
        val_set, split, dataroot, config['pos_class'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        is_train=False,
        seed=config['seed'],
        yaw=yaw
    )

    model = models[config['type']](
        config['gpus'],
        backbone=config['backbone'],
        n_classes=n_classes,
        loss_type=config['loss'],
        weights=weights,
        use_seg=config['seg'],
        use_dep=config['dep'],
        gt=config['gt']
    )

    model.opt = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    if config['mixed']:
        model.scaler = torch.cuda.amp.GradScaler(enabled=True)
        print("Using mixed precision")
    else:
        print("Using full precision")

    if 'pretrained' in config:
        model.load(torch.load(config['pretrained']))
        print(f"Loaded pretrained weights: {config['pretrained']}")
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            model.opt,
            div_factor=10,
            pct_start=.3,
            final_div_factor=10,
            max_lr=config['learning_rate'],
            epochs=config['num_epochs'],
            steps_per_epoch=len(train_loader.dataset) // config['batch_size']
        )

    if 'gamma' in config:
        model.gamma = config['gamma']
        print(f"GAMMA: {model.gamma}")

    if 'ol' in config:
        model.ood_lambda = config['ol']
        print(f"OOD LAMBDA: {model.ood_lambda}")

    if 'k' in config:
        model.k = config['k']

    if 'beta' in config:
        model.beta_lambda = config['beta']
        print(f"Beta lambda is {model.beta_lambda}")

    if 'm_in' in config:
        model.m_in = config['m_in']
    if 'm_out' in config:
        model.m_out = config['m_out']

    if config['seg']:
        os.makedirs(os.path.join(config['logdir'], "seg"), exist_ok=True)

    print("--------------------------------------------------")
    print(f"Using GPUS: {config['gpus']}")
    print(f"Train loader: {len(train_loader.dataset)}")
    print(f"Val loader: {len(val_loader.dataset)}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Output directory: {config['logdir']} ")
    print(f"Using loss {config['loss']}")
    print("--------------------------------------------------")

    writer = SummaryWriter(logdir=config['logdir'])

    writer.add_text("config", str(config))

    step = 0

    # enable to catch errors in loss function
    # torch.autograd.set_detect_anomaly(True)

    for epoch in range(config['num_epochs']):
        model.train()

        writer.add_scalar('train/epoch', epoch, step)

        for images, segs, depths, intrinsics, extrinsics, labels, ood in train_loader:
            t_0 = time()
            ood_loss = None

            outs, seg_outs, loss = model.train_step(images, segs, depths, intrinsics, extrinsics, labels)
            step += 1

            if scheduler is not None:
                scheduler.step()

            if step % 10 == 0:
                print(f"[{epoch}]", step, loss.item(), time()-t_0)
                writer.add_scalar('train/step_time', time() - t_0, step)
                writer.add_scalar('train/loss', loss, step)
                save_pred(model.activate(outs), labels, config['logdir'])

                if ood_loss is not None:
                    writer.add_scalar('train/ood_loss', ood_loss, step)
                    writer.add_scalar('train/id_loss', loss-ood_loss, step)
                    save_unc(model.epistemic(outs), ood, config['logdir'], "epistemic.png", "ood.png")

                if config['seg']:
                    save_pred(model.activate(seg_outs),
                              segs.view(-1, 4, 224, 480), os.path.join(config['logdir'], "seg"))

            if step % 50 == 0:
                iou = get_iou(model.activate(outs).cpu(), labels)

                if config['seg']:
                    seg_iou = get_iou(model.activate(seg_outs).view(-1, 4, 224, 480).cpu(), segs.view(-1, 4, 224, 480))
                    print(f"[{epoch}] {step}", "Seg IOU: ", seg_iou)

                print(f"[{epoch}] {step}", "IOU: ", iou)

                for i in range(0, n_classes):
                    writer.add_scalar(f'train/{classes[i]}_iou', iou[i], step)

        model.eval()

        predictions, seg_preds, ground_truth, seg_labels, oods, aleatoric, epistemic, raw = run_loader(model,
                                                                                                       val_loader)

        if config['seg']:
            seg_iou = get_iou(seg_preds.view(-1, 4, 224, 480).cpu(), seg_labels.view(-1, 4, 224, 480))
            print(f"Validation Seg. mIOU: ", seg_iou)

            for i in range(0, n_classes):
                writer.add_scalar(f'val/seg_{classes[i]}_iou', seg_iou[i], epoch)

        iou = get_iou(predictions, ground_truth)

        for i in range(0, n_classes):
            writer.add_scalar(f'val/{classes[i]}_iou', iou[i], epoch)

        print(f"Validation mIOU: {iou}")

        ood_loss = None

        if config['ood']:
            n_samples = len(raw)
            val_loss = 0
            ood_loss = 0

            for i in range(0, n_samples, config['batch_size']):
                raw_batch = raw[i:i + config['batch_size']].to(model.device)
                ground_truth_batch = ground_truth[i:i + config['batch_size']].to(model.device)
                oods_batch = oods[i:i + config['batch_size']].to(model.device)

                vl, ol = model.loss_ood(raw_batch, ground_truth_batch, oods_batch)

                val_loss += vl
                ood_loss += ol

            val_loss /= (n_samples / config['batch_size'])
            ood_loss /= (n_samples / config['batch_size'])

            writer.add_scalar('val/ood_loss', ood_loss, epoch)
            writer.add_scalar(f"val/loss", val_loss, epoch)
            writer.add_scalar(f"val/uce_loss", val_loss - ood_loss, epoch)

            uncertainty_scores = epistemic[:256].squeeze(1)
            uncertainty_labels = oods[:256].bool()

            fpr, tpr, rec, pr, auroc, aupr, _ = roc_pr(uncertainty_scores, uncertainty_labels)
            writer.add_scalar(f"val/ood_auroc", auroc, epoch)
            writer.add_scalar(f"val/ood_aupr", aupr, epoch)

            print(f"Validation OOD: AUPR={aupr}, AUROC={auroc}")
        else:
            n_samples = len(raw)
            val_loss = 0

            for i in range(0, n_samples, config['batch_size']):
                raw_batch = raw[i:i + config['batch_size']].to(model.device)
                ground_truth_batch = ground_truth[i:i + config['batch_size']].to(model.device)

                vl = model.loss(raw_batch, ground_truth_batch)

                val_loss += vl

            val_loss /= (n_samples / config['batch_size'])

            writer.add_scalar(f"val/loss", val_loss, epoch)

        if ood_loss is not None:
            print(f"Validation loss: {val_loss}, OOD Reg.: {ood_loss}")
        else:
            print(f"Validation loss: {val_loss}")

        model.save(os.path.join(config['logdir'], f'{epoch}.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("config")
    parser.add_argument('-q', '--queue', default=False, action='store_true')
    parser.add_argument('-g', '--gpus', nargs='+', required=False, type=int)
    parser.add_argument('-l', '--logdir', required=False, type=str)
    parser.add_argument('-b', '--batch_size', required=False, type=int)
    parser.add_argument('-s', '--split', default="trainval", required=False, type=str)

    parser.add_argument('--train_set', required=False, type=str)
    parser.add_argument('--val_set', required=False, type=str)

    parser.add_argument('-p', '--pretrained', required=False, type=str)
    parser.add_argument('-o', '--ood', default=False, action='store_true')
    parser.add_argument('-e', '--num_epochs', required=False, type=int)
    parser.add_argument('-c', '--pos_class', default="all", required=False, type=str)
    parser.add_argument('-f', '--fast', default=False, action='store_true')
    parser.add_argument('--seg', default=False, action='store_true')
    parser.add_argument('--dep', default=False, action='store_true')
    parser.add_argument('--gt', default=False, action='store_true')

    parser.add_argument('--seed', default=0, required=False, type=int)
    parser.add_argument('--stable', default=False, action='store_true')

    parser.add_argument('--loss', default="ce", required=False, type=str)
    parser.add_argument('--gamma', required=False, type=float)
    parser.add_argument('--beta', required=False, type=float)
    parser.add_argument('--ol', required=False, type=float)
    parser.add_argument('--k', required=False, type=float)
    parser.add_argument('--m_in', required=False, type=float)
    parser.add_argument('--m_out', required=False, type=float)

    args = parser.parse_args()

    print(f"Using config {args.config}")
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
