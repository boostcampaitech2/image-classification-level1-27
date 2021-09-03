import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from sklearn.model_selection import StratifiedKFold



class CustomDataset(Dataset):
    def __init__(self, args, train=True):
        super().__init__()
        self.data_dir = args.data_dir
        self.info_path = args.info_path
        self.k = args.n_splits              # determine the number of fold
        self.seed = args.seed
        self.k_index= args.k_index          # determine index of k-fold (0 <= k_index < n_splits)
        self.train = train

        self.folders = None
        self.set_k_fold(self.info_path, k_index=self.k_index, k=self.k, seed=self.seed)

        self.input_files = []
        self.images = []
        self.masks = []
        self.genders = []
        self.ages = []
        self.labels = []
        

        self.num_classes = None

        ### prepare images and labels
        if train:
            print("Train Data Loading...")
        else:
            print("Test Data Loading...")

        for directory in tqdm(self.folders):
            image_dir = os.path.join(self.data_dir, directory)
            ID, GENDER, RACE, real_AGE = directory.split('_')

            if GENDER == "male":
                GENDER = 0
            elif GENDER =="female":
                GENDER = 1

            # fix gender label error
            if ID in ['006359', '006360', '006361', '006362', '006363', '006364', '001498-1', '004432']:
                GENDER = 0 if GENDER==1 else 1

            if int(real_AGE) < 30:
                AGE = 0
            elif int(real_AGE) < 60:
                AGE = 1
            else:
                AGE = 2
        
        
            file_list = [f for f in os.listdir(image_dir) if f[0] != '.']
            for file in file_list:
                self.input_files.append(os.path.join(image_dir, file))
                
                if file[0:4] == "mask":
                    MASK = 0
                elif file[0:9] == "incorrect":
                    MASK = 1
                elif file[0:6] == "normal":
                    MASK = 2
                else:
                    raise

                self.images.append(np.array(Image.open(os.path.join(image_dir, file))))    
                self.masks.append(MASK)
                self.genders.append(GENDER)
                self.ages.append(AGE)
                self.labels.append(MASK*6 + GENDER*3 + AGE)

        if self.train:
            weight_count = [0]*18
            for c in self.labels:
                weight_count[c] += 1
            self.class_weight = [1/i for i in weight_count]
                    
    def __len__(self):
        return len(self.labels)
    
    
    def __getitem__(self, idx):
        ### load image
        image = self.images[idx]
        if self.transform:
            image = self.transform(image=image)['image']
        image.type(torch.float32)
        
        ### load label
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        
        return image, label

    def set_k_fold(self, info_path, k_index=None, k=5, seed=1997):
        """
            output: train_folder, valid_folder
        """

        if not k_index in range(k):
            raise Exception('n_splits에 맞는 index를 입력해주세요')

        train_info = pd.read_csv(info_path)

        ### age/gender 동일 비율로 K Fold진행
        new_age = np.array(train_info['age'])
        new_gender = np.array(train_info['gender'])
        str_for_split = new_age.astype(str)+new_gender

        SFK = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        for idx, (train_index, valid_index) in enumerate(SFK.split(train_info, str_for_split)):
            if idx == k_index:
                self.folders = train_info['path'][train_index] if self.train else train_info['path'][valid_index]
                break

    def set_transform(self, transform):
        self.transform = transform

class CustomTestDataset(Dataset):
    def __init__(self, data_path, train=False):
        super(CustomTestDataset, self).__init__()
        self.data_dir = data_path
        self.train = train   
        self.image = []
        for file in tqdm(self.data_dir):
            self.image.append( np.array(Image.open(file)))

                    
    def __len__(self):
        return len(self.data_dir)
    
    def __getitem__(self, idx):
        ### load image
        image = self.image[idx]
        if self.transform:
            image = self.transform(image=image)['image']
        image.type(torch.float32)
        
        return image

    def set_transform(self, transform):
        self.transform = transform




