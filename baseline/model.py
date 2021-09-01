import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from efficientnet_pytorch import EfficientNet


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class CustomModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features

        ### fc layer 변경
        self.backbone.fc = nn.Identity()
        self.mask_fc = nn.Linear(in_features, 3)
        self.gender_fc = nn.Linear(in_features, 1)
        self.age_fc = nn.Linear(in_features, 3)
        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        self.num_classes = None

    def forward(self, x):
        x = self.backbone(x)
        mask_output = self.mask_fc(x)
        gender_output = self.gender_fc(x)
        age_output = self.age_fc(x)
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return mask_output, gender_output, age_output


class CustomModel_E(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b3')
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()
        self.mask_fc = nn.Linear(in_features, 3)
        self.gender_fc = nn.Linear(in_features, 1)
        self.age_fc = nn.Linear(in_features, 3)
        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        self.num_classes = None

    def forward(self, x):
        x = self.backbone(x)
        mask_output = self.mask_fc(x)
        gender_output = self.gender_fc(x)
        age_output = self.age_fc(x)
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return mask_output, gender_output, age_output