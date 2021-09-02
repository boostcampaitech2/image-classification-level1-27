import argparse
import os
from importlib import import_module
import multiprocessing
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from dataset import MaskBaseDataset
from tqdm import tqdm
from transform import get_tta_transform
from utils import *


def load_model(saved_model, device):
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(
    ).to(device)
    model_path = os.path.join(saved_model, 'best_score.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    if args.tta :
        args.batch_size = 1
        tta_transforms = get_tta_transform()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = load_model(model_dir, device).to(device)
    
    model.eval()
    img_root = os.path.join(data_dir, 'crop_images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)
    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]

    #dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)
    
    test_set = dataset_module(
        data_path = img_paths,
        train=False)

    transform_module = getattr(import_module("transform"), args.augmentation)  # default: BaseAugmentation

    test_transform = transform_module(
        train=False
    )
      
    test_set.set_transform(test_transform)

    loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    ans = []
    with torch.no_grad():
        for idx, images in enumerate(tqdm(loader)):
            images = images.to(device)

            if args.tta:
                m_outs, g_outs, a_outs = tta(tta_transforms, model, images)
                tta_m_preds = torch.unsqueeze(torch.argmax(m_outs, dim=-1),0).cpu()
                tta_a_preds = torch.unsqueeze(torch.argmax(a_outs, dim=-1),0).cpu()
                if g_outs >= 0.5 : tta_g_preds = 1
                else: tta_g_preds= 0
                tta_g_preds = torch.unsqueeze(torch.tensor(tta_g_preds),0).cpu()
                tta_preds = label_encoder(tta_m_preds, tta_g_preds, tta_a_preds)
                ans.append(tta_preds[0].item())
            else:
                model.eval()
                m_outs, g_outs, a_outs = model(images)
                m_preds = torch.argmax(m_outs, dim=-1).cpu()
                g_preds = (g_outs>0).squeeze().cpu()
                a_preds = torch.argmax(a_outs, dim=-1).cpu()
                pred = label_encoder(m_preds, g_preds, a_preds)
                
                # pred = pred.argmax(dim=-1)
                ans.extend(pred.cpu().numpy())

    info['ans'] = ans
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--dataset', type=str, default='CustomTestDataset', help='dataset augmentation type (default: CustomDataset)')
    parser.add_argument('--tta', type=bool, default=False, help='dataset augmentation type (default: CustomDataset)')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='CustomModel', help='model type (default: BaseModel)')
    parser.add_argument('--resize', type=tuple, default=(512, 384), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--augmentation', type=str, default='Augmentation_384', help='data augmentation type (default: BaseAugmentation)')    
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/ensemble'))  # modified by ihyun
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
