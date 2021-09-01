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

from utils import *


def load_model(saved_model, device):
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(
    ).to(device)
    model_path = os.path.join(saved_model, 'best_acc.pth')
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
    m_out_list = []
    g_out_list = []
    a_out_list = []
    with torch.no_grad():
        for idx, images in enumerate(tqdm(loader)):
            images = images.to(device)


            m_outs, g_outs, a_outs = model(images)
            
            m_outs = nn.functional.softmax(m_outs).cpu()
            g_outs = torch.sigmoid(g_outs).cpu()
            a_outs = nn.functional.softmax(a_outs).cpu()
            
            m_out_list += [m_outs]
            g_out_list += [g_outs]
            a_out_list += [a_outs]


            
            # out = out.argmax(dim=-1)
        m_outs = torch.cat(m_out_list,0)
        g_outs = torch.cat(g_out_list,0)
        a_outs = torch.cat(a_out_list,0)

        print(m_outs.shape)
        print(g_outs.shape)
        print(a_outs.shape)

    info['m_out_0'] = m_outs[:, 0]
    info['m_out_1'] = m_outs[:, 1]
    info['m_out_2'] = m_outs[:, 2]
    
    info['g_out'] = g_outs[:,0]

    info['a_out_0'] = a_outs[:, 0]
    info['a_out_1'] = a_outs[:, 1]
    info['a_out_2'] = a_outs[:, 2]
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--dataset', type=str, default='CustomTestDataset', help='dataset augmentation type (default: CustomDataset)')
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
