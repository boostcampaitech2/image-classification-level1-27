import numpy as np
import argparse
from sklearn.metrics import f1_score

def label_encoder(m_labels, g_labels, a_labels):
    return m_labels*6+ g_labels*3+ a_labels

def label_decoder(labels):
    return labels//6, labels%6//3, labels%6%3

def label_encoder_mg(m_labels, g_labels):
    return m_labels*6+ g_labels*3

def label_decoder_mg(labels):
    return labels//6, labels%6//3



def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

def get_f1_score(gt, pr, verbose=False, detail=False):
    m_gt, g_gt, a_gt = label_decoder(gt)
    m_pr, g_pr, a_pr = label_decoder(pr)
    
    score = dict()

    score['mask'] = [] 
    for i in range(3):
        tmp_gt = (m_gt==i)
        tmp_pr = (m_pr==i)
        score['mask'].append(f1_score(tmp_gt, tmp_pr, average='macro'))
    
    
    score['gender'] = [] 
    for i in range(2):
        tmp_gt = (g_gt==i)
        tmp_pr = (g_pr==i)
        score['gender'].append(f1_score(tmp_gt, tmp_pr, average='macro'))

    
    score['age'] = [] 
    for i in range(3):
        tmp_gt = (a_gt==i)
        tmp_pr = (a_pr==i)
        score['age'].append(f1_score(tmp_gt, tmp_pr, average='macro'))

    score['total'] = f1_score(gt, pr, average='macro')
    if verbose:

        print(f"===========f1_score===========")
        print(f"label\t  0\t  1\t  2")
        print(f"mask\t{score['mask'][0]:.4}\t{score['mask'][1]:.4}\t{score['mask'][2]:.4}")
        print(f"gender\t{score['gender'][0]:.4}\t{score['gender'][1]:.4}")
        print(f"age\t{score['age'][0]:.4}\t{score['age'][1]:.4}\t{score['age'][2]:.4}")
        print(f"============{score['total']:.4}============")

    if detail:
        all_f1 = []
        for i in range(18):
            tmp_gt = (gt==i)
            tmp_pr = (pr==i)
            all_f1.append(f1_score(tmp_gt, tmp_pr))
            print(f"==============male==============\t============female============")
            print(f" \twear\t  \tnot_wear\t\twear\t \tnot_wear\t")
            print(f"  ~30\t{all_f1[0]:.4}\t{all_f1[6]:.4}\t{all_f1[12]:.4}\t\t  ~30\t{all_f1[3]:.4}\t{all_f1[9]:.4}\t{all_f1[15]:.4}")
            print(f"30~60\t{all_f1[1]:.4}\t{all_f1[7]:.4}\t{all_f1[13]:.4}\t\t30~60\t{all_f1[4]:.4}\t{all_f1[10]:.4}\t{all_f1[16]:.4}")
            print(f"60~  \t{all_f1[2]:.4}\t{all_f1[8]:.4}\t{all_f1[14]:.4}\t\t30~60\t{all_f1[5]:.4}\t{all_f1[11]:.4}\t{all_f1[17]:.4}")
            print(f"============={score['total']:.3}=============\t============={score['total']:.3}=============")

    return score      


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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value esxpected.')
