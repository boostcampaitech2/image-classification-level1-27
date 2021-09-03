import argparse
import os
from importlib import import_module
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils import *
from transform import get_tta_transform


def load_model(saved_model, device):
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(
    ).to(device)
    model_path = os.path.join(saved_model, 'best_score.pth')
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

    transform_module = getattr(import_module("transform"), args.augmentation)  # default: BaseAugmentation

    test_transform = transform_module(
        train=False
    )
      
    test_set.set_transform(test_transform)

    loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=1,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    # --- get_tta_transform \ default : [horizontal_flip,]
    tta_transforms = get_tta_transform()


    print("Calculating inference results..")
    predictions = []
    with torch.no_grad():
        for images in loader:
            tta_soft_voting = []    
            images = images.to(device)
            origin_output = model(images)
            origin_vote = nn.functional.softmax(origin_output, -1)
            tta_soft_voting.append(origin_vote)

            for transform in tta_transforms:
                tta_images = transform.augment_image(images)
                tta_output = model(tta_images)
                tta_vote = nn.functional.softmax(tta_output, -1)
                tta_soft_voting.append(tta_vote)
            
            tta_soft_voting = torch.stack(tta_soft_voting).mean(0)

            tta_predict = tta_soft_voting.argmax(dim=-1)        
            predictions.extend(tta_predict.cpu().numpy())

    info['ans'] = predictions
    info.to_csv(os.path.join(output_dir, f'output.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--dataset', type=str, default='CustomTestDataset', help='dataset augmentation type (default: CustomTestDataset)')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for validing (default: 64)')
    parser.add_argument('--model', type=str, default='CustomModel_Arc', help='model type (default: CustomModel_Arc)')
    parser.add_argument('--augmentation', type=str, default='Augmentation_384', help='data augmentation type (default: Augmentation_384)')    
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/best'))  # modified by ihyun
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    if not os.path.isdir('/opt/ml/input/data/train/crop_images/') or not os.path.isdir('/opt/ml/input/data/eval/crop_images/'):
        from create_crop_images import create_crop_images
        create_crop_images()

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
