# image-classification-level1-27

## Project
- 성별, 나이, 마스크 착용 여부가 labeling 되어있는 Asian face data 18900개를 학습해 12600개의 데이터 예측하기
- 성별은 40 대 60으로 여성이 더 많았음.
- 나이는 30대 미만, 30대 이상 60대 미만, 60대 이상으로 3개의 클래스로 구성 되어 있었음.
- 마스크 착용 여부는 한 사람당 각기 다른 종류의 마스크를 올바르게 착용한 사진 5장, 잘못 착용한 사진 1장, 착용하지 않은 사진 1장으로 7장씩 구성되어 있었음.



## Result
- Private F1_score: 0.715



## Getting Started
```bash
pip install -r baseline/requirements.txt
```



### Training
```bash
cd baseline
python train.py
```
- crop 이미지를 사용하기 때문에 데이터 없을 시 자동 생성함. (절대 경로)



### Inference
```
python inference.py
```



## Code Structure
```
├── baseline                       # our code
│   ├── train.py                # to train your data
│   ├── inferenc.py             # to inference 
│   ├── model.py
│   ├── loss.py
│   ├── transform.py
│   ├── utils.py
│   └── create_crop_images.py   # to create crop image
└── crop_data                   # crop box information
```



## Model
- imagenet pretrained resnet18을 사용하였고, <a href = 'https://github.com/ronghuaiyang/arcface-pytorch'>arcface loss</a>를 사용하기 위해 마지막 fc layer을 바꿔 주었다.
![Original-ResNet-18-Architecture](https://user-images.githubusercontent.com/78612464/132025698-63850b47-bad8-4007-a023-b5211d4fbe5f.png)  
<img width="235" alt="arc_face" src="https://user-images.githubusercontent.com/78612464/132025759-239afd7f-761d-4e5d-bc05-a4ae7135e1c7.png">



## Dataset
- 사람의 얼굴에 더 집중하기 위해 이미지를 crop하여 사용하였다
- <a href = 'https://github.com/rakshit087/Face-Mask-Detection-PyTorch'>1차 crop</a>: 마스크를 착용한 사람에서 face를 잘 detect하지 못하는 경우가 많았다. 그래서 우리 조는 마스크를 착용한 데이터셋을 사용해 학습한 모델을 사용하여 1차적으로 이미지를 크롭하였다. 18900개중 539개를 제외하고 crop했다. 특히 마스크를 잘못 착용한 데이터에서 얼굴 검출이 잘 안되었다.
- <a href = 'https://github.com/biubug6/Pytorch_Retinaface'>2차 crop</a>: face crop에 성능이 좋다고 알려진 Retinaface를 이용해 2차 crop을 해주었고, box가 너무 작게 예측된(잘못 예측된) 것들을 걸러내니 2장이 남아 직접 box labeling하여 사용하였다.



## Train detail
- arcface loss를 이용하였을 때 훨씬 더 빠른 학습 속도를 보였고, margin parameter는 0.4를 주었다.
- arcface loss와 함께 사용할 다양한 loss(eg. BCE, focal, f1 loss ...)를 실험을 해보았는데, weight 주지 않고, label smoothing을 주지 않은 focal loss가 가장 성능이 좋아 focal loss를 사용하였다.($\gamma$=2)
- trainset overfitting이 SGD를 사용하였을 때 안정적이어서 SGD를 사용했다. (momentum=0.9, weight_decay=5e-4)
- <a href = 'https://github.com/qubvel/ttach'>TTA</a>: 다양한 transform을 적용하여 실험을 진행해보았다. 데이터 자체가 워낙 정형화 되어있는 탓인지 horizontal flip만 준게 가장 성능이 좋았다.

