from re import A
import numpy as np
from sklearn.metrics import f1_score
import os
import torch
from torch import nn

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