# save crop images

import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm

def resize_box(s_X, s_Y, e_X, e_Y, ratio=4/3, scale=1):
    center_X, center_Y = (s_X+e_X)/2, (s_Y+e_Y)/2
    
    width, height = e_X-s_X, s_Y-e_Y
    
    height = max(width*ratio, height)*scale
    width = height/ratio
    
    s_X = max(int(center_X - width/2),0)
    e_X = min(int(center_X + width/2),512)
    s_Y = max(int(center_Y - height/2),0)
    e_Y = min(int(center_Y + height/2),512)
    
    return s_X, s_Y, e_X, e_Y

def save_train_crop_image(crop_info, save_root, scale=1.2):
    for path, s_X, s_Y, e_X, e_Y in tqdm(np.array(crop_info)):
        folder, file = path.split('/')[-2:]
        
        s_X, s_Y, e_X, e_Y = resize_box(s_X, s_Y, e_X, e_Y, scale=scale)

        save_dir = os.path.join(save_root, folder)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, file)

        before_crop = cv2.imread(path)
        after_crop = before_crop[s_Y:e_Y,s_X:e_X]
        
        
        after_resize = cv2.resize(after_crop, dsize=(288, 384), interpolation=cv2.INTER_LINEAR)
        
        if not cv2.imwrite(save_path, after_resize):
            print(f"실패: {file}")

def save_test_crop_image(crop_info, save_root, scale=1.2):
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    for path, s_X, s_Y, e_X, e_Y in tqdm(np.array(crop_info)):
        file = path.split('/')[-1]
        
        s_X, s_Y, e_X, e_Y = resize_box(s_X, s_Y, e_X, e_Y, scale=scale)

        save_path = os.path.join(save_root, file)
        
        before_crop = cv2.imread(path)
        after_crop = before_crop[s_Y:e_Y,s_X:e_X]
        
        
        after_resize = cv2.resize(after_crop, dsize=(288, 384), interpolation=cv2.INTER_LINEAR)

        if not cv2.imwrite(save_path, after_resize):
            print(f"실패 {file}")

def create_crop_images():
    train_crop_info = pd.read_csv("../crop_data/total_train_crop.csv")
    train_save_root = '/opt/ml/input/data/train/crop_images/'
    print("Saving train crop images...")
    save_train_crop_image(train_crop_info, train_save_root, scale=1.2)

    test_crop_info = pd.read_csv("../crop_data/total_test_crop.csv")
    test_save_root = '/opt/ml/input/data/eval/crop_images/'
    print("Saving test crop images...")
    save_test_crop_image(test_crop_info, test_save_root, scale=1.2)

if __name__ == '__main__':
    create_crop_images()