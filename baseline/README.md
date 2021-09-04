# image-classification-level1-27

## Project 개요
- 목표 : 사람의 이미지로 성별, 나이, 마스크 착용 여부 예측
  - 각 클래스에 따른 총 18개의 경우의 수 예측
  - 성별(남,여), 나이(30대 미만, 30대 이상 60대미만, 60대 이상), 마스크 착용여부(미착용, 잘못된 착용, 올바른 착용)
- Data
  - Training Data : labeling 되어있는 Asian face data 18900개
  - Test Data : Asian face data 12600개
- 분포 
  - 성별은 40 대 60으로 여성이 더 많았음.
  - 나이대는 30대 미만이 가장 많았으며 60 대이상이 가장 적었다. 둘 사이의 빈도수 편차가 큰 편임
  - 마스크 착용 여부는 한 사람당 각기 다른 종류의 마스크를 올바르게 착용한 사진 5장, 잘못 착용한 사진 1장, 착용하지 않은 사진 1장으로 7장씩 구성되어 있었음.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Advanced Examples](#advanced-Examples)
3. [Code Structure](#code-structure)
4. [Detail](#detail)
5. [Contributor](#contributor)


### Result
- Private F1_score: 0.715



## Getting Started
```bash
pip install -r baseline/requirements.txt
```

### Quick start
```bash
cd baseline
python train.py
```
- crop 이미지를 사용하기 때문에 데이터 없을 시 자동 생성함. (절대 경로)



### Inference
```
python inference.py
```


## Advanced Examples
### Advanced Train
```
python train.py --batch_size 32 --seed 2021 --optimizer Adam --lr 1e-3 ...
```
### Train Arguments
| Argument        | DataType    | Default          | Help                          |
|-----------------|:-----------:|:----------------:|:-----------------------------:|
| seed            | int         | 1997             | random seed                   |
| epochs          | int         | 40               | number of epochs to train     |
| dataset         | str         | CustomDataset    | dataset augmentation type     |
| augmentation    | str         | Augmentation_384 | data augmentation type        |
| batch_size      | int         | 1997             | input batch size for training |
| valid_batch_size| int         | 32               | input batch size for validing |
| model           | str         | CustomModel_Arc  | model type                    |
| optimizer       | str         | SGD              | optimizer type                |
| scheduler       | str         | reducelr         | scheduler type                |
| lr              | float       | 3e-3             | learning rate                 |
| criterion       | str         | focal            | criterion type                |
| lr_decay_step   | int         | 20               | learning rate scheduler deacy step |
| log_interval    | int         | 20               | how many batches to wait before logging training status |
| name            | str         | resnet18_arc     | dir name for trained model    |
| n_splits        | int         | 5                | number for K-Fold validation  |
| k_index         | int         | 4                | number of K-Fold validation   |
| data_dir        | str         | _                | image data path               |
| model_dir       | str         | _                | model saving path             |
| arc_scale       | float       | 30.0             | arcface scale                 |
| arc_margin      | float       | 0.4              | arcface margin                |

### Advanced inference
```
python inference.py --batch_size 32 ...
```

### inference Arguments
| Argument        | DataType    | Default          | Help                          |
|-----------------|:-----------:|:----------------:|:-----------------------------:|
| dataset         | str         | CustomTestDataset | dataset augmentation type    |
| batch_size      | int         | 64               | input batch size for validing |
| model           | str         | CustomModel_Arc  | model type                    |
| augmentation    | str         | Augmentation_384 | data augmentation type        |
| data_dir        | str         | _                | image data path               |
| model_dir       | str         | _                | model loading path            |
| output_dir      | str         | _                | inference file saving path    |


## Code Structure
```
├── baseline                       # our code
│   ├── train.py                # to train your data
│   ├── inference.py            # to inference 
│   ├── model.py                # our CustomModel    
│   ├── loss.py                  
│   ├── dataset.py
│   ├── transform.py
│   ├── utils.py
│   └── create_crop_images.py   # to create crop image
└── crop_data                   # crop box information
```


## Detail
---
### Model
- imagenet pretrained resnet18을 사용하였고, <a href = 'https://github.com/ronghuaiyang/arcface-pytorch'>arcface loss</a>를 사용하기 위해 마지막 fc layer을 바꿔 주었다.
![Original-ResNet-18-Architecture](https://user-images.githubusercontent.com/78612464/132025698-63850b47-bad8-4007-a023-b5211d4fbe5f.png)  
<img width="235" alt="arc_face" src="https://user-images.githubusercontent.com/78612464/132025759-239afd7f-761d-4e5d-bc05-a4ae7135e1c7.png">



### Dataset
- 사람의 얼굴에 더 집중하기 위해 이미지를 crop하여 사용하였다
- <a href = 'https://github.com/rakshit087/Face-Mask-Detection-PyTorch'>1차 crop</a>: 마스크를 착용한 사람에서 face를 잘 detect하지 못하는 경우가 많았다. 그래서 우리 조는 마스크를 착용한 데이터셋을 사용해 학습한 모델을 사용하여 1차적으로 이미지를 크롭하였다. 18900개중 539개를 제외하고 crop했다. 특히 마스크를 잘못 착용한 데이터에서 얼굴 검출이 잘 안되었다.
- <a href = 'https://github.com/biubug6/Pytorch_Retinaface'>2차 crop</a>: face crop에 성능이 좋다고 알려진 Retinaface를 이용해 2차 crop을 해주었고, box가 너무 작게 예측된(잘못 예측된) 것들을 걸러내니 2장이 남아 직접 box labeling하여 사용하였다.



### Train detail
- arcface loss를 이용하였을 때 훨씬 더 빠른 학습 속도를 보였고, margin parameter는 0.4를 주었다.
- arcface loss와 함께 사용할 다양한 loss(eg. BCE, focal, f1 loss ...)를 실험을 해보았는데, weight 주지 않고, label smoothing을 주지 않은 focal loss가 가장 성능이 좋아 focal loss를 사용하였다.($\gamma$=2)
- trainset overfitting이 SGD를 사용하였을 때 안정적이어서 SGD를 사용했다. (momentum=0.9, weight_decay=5e-4)
- <a href = 'https://github.com/qubvel/ttach'>TTA</a>: 다양한 transform을 적용하여 실험을 진행해보았다. 데이터 자체가 워낙 정형화 되어있는 탓인지 horizontal flip만 준게 가장 성능이 좋았다.

## Contributor
- 김인재([github](https://github.com/K-nowing)) : 팀 리더, Baseline 구성 및 성능 향상 
- 송민재([github](https://github.com/Jjackson-dev)): 각 모델과 Augmentation 실험 등 성능향상 
- 윤하정([github](https://github.com/YHaJung)): 협업 규칙정의, 모델실험
- 송이현([blog](https://hello-sarah.tistory.com/)): 데이터 수집, 각종 성능 실험
- 박상류([github](https://github.com/psrpsj)): 모델 구조, Hyperparameter 실험
- 채유리([github](https://github.com/yoorichae)): 모델, Augmentation 실험



