import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
import math
from sklearn.metrics import f1_score
import os

def label_encoder(m_labels, g_labels, a_labels):
    return m_labels*6+ g_labels*3+ a_labels

def label_decoder(labels):
    return labels//6, labels%6//3, labels%6%3

def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

def get_f1_score(gt, pr, verbose=False):
    m_gt, g_gt, a_gt = label_decoder(gt)
    m_pr, g_pr, a_pr = label_decoder(pr)
    
    score = dict()
    score['mask'], score['age'],score['gender'] = [], [], []
    for i in range(3):
        if i<3 :
            gender_gt, gender_pr = (g_gt==i), (g_pr==i)
            score['gender'].append(f1_score(gender_gt, gender_pr, average='macro'))
        mask_gt, mask_pr = (m_gt==i), (m_pr==i)
        age_gt, age_pr = (a_gt==i), (a_pr==i)
        score['age'].append(f1_score(age_gt, age_pr, average='macro'))
        score['mask'].append(f1_score(mask_gt, mask_pr, average='macro'))

    score['total'] = f1_score(gt, pr, average='macro')
    if verbose:
        print(f"===========f1_score===========")
        print(f"label\t  0\t  1\t  2")
        print(f"mask\t{score['mask'][0]:.4}\t{score['mask'][1]:.4}\t{score['mask'][2]:.4}")
        print(f"gender\t{score['gender'][0]:.4}\t{score['gender'][1]:.4}")
        print(f"age\t{score['age'][0]:.4}\t{score['age'][1]:.4}\t{score['age'][2]:.4}")
        print(f"============{score['total']:.4}============")
    return score      

 
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
 

def rand_bbox(size, lam): # size : [Batch_size, Channel, Width, Height]
    W = size[2] 
    H = size[3] 
    cut_rat = lam  # 패치 크기 비율
    cut_h = np.int(H * cut_rat)  

   	# 패치의 중앙 좌표 값 cx, cy
    # cy = np.random.randint(H)
		
    # 패치 모서리 좌표 값 
    bbx1 = 0
    bbx2 = W
    if np.random.random() > 0.5:
        bby1 = 0
        bby2 = int(cut_h)
    else:
        bby1 = H-int(cut_h)
        bby2 = H
   
    return bbx1, bby1, bbx2, bby2


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.30, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        if label is None:
            return self.inference(input)

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

    def inference(self, input):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))


        return cosine
    
def tta(tta_transforms, model, inputs):
    m_out_list = [[],[],[]]
    g_out_list = []
    a_out_list = [[],[],[]]
    for transformer in tta_transforms: # custom transforms or e.g. tta.aliases.d4_transform() 
                    # augment image
        augmented_image = transformer.augment_image(inputs)
        m, g, a = model(augmented_image)
                    
        m_tensor = nn.functional.softmax(m).cpu()[0]
        g_tensor = torch.sigmoid(g).cpu()[0]
        a_tensor = nn.functional.softmax(a).cpu()[0]
                    # save results
        m_out_list[0].append(m_tensor[0])
        m_out_list[1].append(m_tensor[1])
        m_out_list[2].append(m_tensor[2])
        g_out_list.append(g_tensor)
        a_out_list[0].append(a_tensor[0])
        a_out_list[1].append(a_tensor[1])
        a_out_list[2].append(a_tensor[2])
                    
                # reduce results as you want, e.g mean/max/min

    m1_result = torch.mean(torch.tensor(m_out_list[0]))
    m2_result = torch.mean(torch.tensor(m_out_list[1]))
    m3_result = torch.mean(torch.tensor(m_out_list[2]))
    a1_result = torch.mean(torch.tensor(a_out_list[0]))
    a2_result = torch.mean(torch.tensor(a_out_list[1]))
    a3_result = torch.mean(torch.tensor(a_out_list[2]))
    g_result = torch.mean(torch.tensor(g_out_list))

    m_outs = torch.tensor([m1_result, m2_result, m3_result])
    a_outs = torch.tensor([a1_result, a2_result, a3_result])

    return m_outs, torch.tensor(g_result).cpu(), a_outs