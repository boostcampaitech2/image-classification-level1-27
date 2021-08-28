import numpy as np
from sklearn.metrics import f1_score

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

def get_f1_score(gt, pr):
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
    return score      