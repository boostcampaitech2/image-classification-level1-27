# python [train.py](http://train.py/) --name baseline_augpp --model CustomModel --augmentation Augmentation_384 --lr 3e-3  —k_index 999

import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion
from utils import *

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.sample(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"




T1 = 50
T2 = 100
af = 0.8

def alpha_weight(step):
    if step < T1:
        return 0.0
    elif step > T2:
        return af
    else:
         return ((step-T1) / (T2-T1))*af



def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: BaseAugmentation
    # dataset = dataset_module(
    #     data_dir=data_dir,
    #     info_path=args.info_path
    # )
    # num_classes = dataset.num_classes  # 18
    train_set = dataset_module(
        args=args,
        train=True
    )
    val_set = dataset_module(
        args=args,
        train=False
    )


    img_root = os.path.join('/opt/ml/input/data/eval', 'crop_images')
    info_path = os.path.join('/opt/ml/input/data/eval', 'info.csv')
    info = pd.read_csv(info_path)
    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]

    test_dataset_module = getattr(import_module("dataset"), 'CustomTestDataset')
    test_set = test_dataset_module(
        data_path = img_paths,
        train=False
    )


    # -- augmentation
    # transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    # transform = transform_module(
    #     resize=args.resize,
    #     mean=dataset.mean,
    #     std=dataset.std,
    # )
    transform_module = getattr(import_module("transform"), args.augmentation)  # default: BaseAugmentation
    train_transform = transform_module(
        train=True
    )    
    val_transform = transform_module(
        train=False
    )
    train_set.set_transform(train_transform)    
    val_set.set_transform(val_transform)
    test_set.set_transform(val_transform)
    # -- data_loader
    # train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size//2,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size//2,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )
    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    # model = model_module(
    #     num_classes=num_classes
    # ).to(device)
    model = model_module(
    ).to(device)
    #model.load_state_dict(torch.load("./model/ensemble4/best_acc.pth", map_location=device))
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    # criterion = create_criterion(args.criterion)  # default: cross_entropy
    criterion_mask = create_criterion(args.criterion_mask)  # default: cross_entropy
    criterion_gender = create_criterion(args.criterion_gender)  # default: bce_loss
    criterion_age = create_criterion(args.criterion_age)  # default: cross_entropy

    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        #momentum=0.9,
        weight_decay=5e-4
    )
    #scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
    if args.scheduler == 'reducelr':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=0.002, min_lr=1e-4)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=2e-4)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_score = 0
    best_val_loss = np.inf
    step = 0
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        
        for idx, (train_batch, test_batch) in enumerate(zip(train_loader, test_loader)):

            model.eval()
            test_inputs = test_batch
            test_inputs = test_inputs.to(device)
            outputs_unlabeled = model(test_inputs)
            t_m_labels = torch.argmax(outputs_unlabeled[0], dim=-1)
            t_g_labels = (outputs_unlabeled[1]>0).type(torch.float32)
            t_a_labels = torch.argmax(outputs_unlabeled[2], dim=-1)

            model.train()

            

            inputs, labels = train_batch
            inputs = inputs.to(device)
            m_labels = labels['mask'].to(device)
            g_labels = labels['gender'].to(device)
            a_labels = labels['age'].to(device)

            optimizer.zero_grad()
            
            t_m_outs, t_g_outs, t_a_outs = model(test_inputs)
            t_m_loss = criterion_mask(t_m_outs, t_m_labels)
            t_g_loss = criterion_gender(t_g_outs, t_g_labels)
            t_a_loss = criterion_age(t_a_outs, t_a_labels)
            t_loss = t_m_loss + t_g_loss + t_a_loss

            m_outs, g_outs, a_outs = model(inputs)
            m_loss = criterion_mask(m_outs, m_labels)
            g_loss = criterion_gender(g_outs, g_labels)
            a_loss = criterion_age(a_outs, a_labels)
            o_loss = m_loss + g_loss + a_loss

            loss = o_loss + t_loss*alpha_weight(step)
            if (idx+1)%50 == 0:
                step += 1
                if step%10 == 0:
                    print(f"STEP {step}")

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            with torch.no_grad():
                m_preds = torch.argmax(m_outs, dim=-1)
                g_preds = (g_outs>0).squeeze()
                a_preds = torch.argmax(a_outs, dim=-1)
                preds = label_encoder(m_preds, g_preds, a_preds)
                labels = label_encoder(m_labels, g_labels.squeeze(), a_labels)
                matches += (preds == labels).sum().item()

            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0

        #scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            val_predicts = torch.empty(0)
            val_targets = torch.empty(0)
            figure = None

            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                m_labels = labels['mask'].to(device)
                g_labels = labels['gender'].to(device)
                a_labels = labels['age'].to(device)

                ### update val_predicts & val_targets
                m_outs, g_outs, a_outs = model(inputs)
                m_preds = torch.argmax(m_outs, dim=-1).cpu()
                g_preds = (g_outs>0).squeeze().cpu()
                a_preds = torch.argmax(a_outs, dim=-1).cpu()
                preds = label_encoder(m_preds, g_preds, a_preds)
                labels = label_encoder(m_labels.cpu(), g_labels.cpu().squeeze(), a_labels.cpu())
                val_predicts = torch.cat((val_predicts,preds))
                val_targets = torch.cat((val_targets,labels))

                m_loss = criterion_mask(m_outs, m_labels).item()
                g_loss = criterion_gender(g_outs, g_labels).item()
                a_loss = criterion_age(a_outs, a_labels).item()
                loss_item = (m_loss + g_loss + a_loss)
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = denormalize_image(inputs_np, val_transform.mean, val_transform.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc:
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best_acc.pth")
                best_val_acc = val_acc
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")


            ### print f1_score
            score = get_f1_score(val_targets, val_predicts, verbose=True)
            val_score = score['total']
            if val_score > best_val_score:
                print(f"New best model for f1 score : {val_score:4.2}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best_score.pth")
                best_val_score = val_score
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best score : {best_val_score:4.2%}, best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_scalar("Val/f1_score", val_score, epoch)
            logger.add_figure("results", figure, epoch)
            print()

        if args.scheduler == 'reducelr':
            scheduler.step(torch.tensor(val_loss))
        elif args.scheduler == 'cosine':
            scheduler.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    from dotenv import load_dotenv
    import os
    load_dotenv(verbose=True)



    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=1997, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='CustomDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='Augmentation_384', help='data augmentation type (default: BaseAugmentation)')
    # parser.add_argument("--resize", nargs="+", type=list, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=32, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='CustomModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: SGD)')
    parser.add_argument('--scheduler', type=str, default='reducelr', help='scheduler type (default: reducelr)')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    # parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--criterion_mask', type=str, default='cross_entropy', help='criterion_mask type (default: cross_entropy)')
    parser.add_argument('--criterion_gender', type=str, default='bce_loss', help='criterion_gender type (default: bce_loss)')
    parser.add_argument('--criterion_age', type=str, default='cross_entropy', help='criterion_age type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='pseudo_4', help='model save at {SM_MODEL_DIR}/{name}')

    # Dataset
    parser.add_argument('--n_splits', type=int, default=5, help='number for K-Fold validation')
    parser.add_argument('--k_index', type=int, help='number of K-Fold validation')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/crop_images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--info_path', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/train.csv'))

    parser.add_argument('--drop_size', type=int, default=32, help='aug drop size')
    args = parser.parse_args()
    print(args)
    
    data_dir = args.data_dir
    model_dir = args.model_dir

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)


    train(data_dir, model_dir, args)