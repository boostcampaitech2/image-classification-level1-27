import argparse
import os
from importlib import import_module
import multiprocessing
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset

from utils import *


def load_model(saved_model, device):
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(
    ).to(device)
    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
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
        )

    loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count()//2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            # pred = model(images)

            m_outs, g_outs, a_outs = model(images)
            m_preds = torch.argmax(m_outs, dim=-1).cpu()
            g_preds = (g_outs>0).squeeze().cpu()
            a_preds = torch.argmax(a_outs, dim=-1).cpu()
            pred = label_encoder(m_preds, g_preds, a_preds)
            
            # pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--dataset', type=str, default='CustomTestDataset', help='dataset augmentation type (default: CustomDataset)')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='CustomModel', help='model type (default: BaseModel)')
    parser.add_argument('--resize', type=tuple, default=(384, 288), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')    
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', '/opt/ml/model/exp8'))  # modified by ihyun
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
