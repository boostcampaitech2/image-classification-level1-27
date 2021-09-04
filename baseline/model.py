import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import Parameter
import math


# --- imagenet pretrained resnet18 with ArcFace loss
class CustomModel_Arc(nn.Module):
    def __init__(self, scale=30.0, margin=0.5, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Identity()
        self.metric_fc = ArcMarginProduct(in_features, 18, scale=scale, margin=margin)
        self.num_classes = None

    def forward(self, x, label=None):
        feature = self.backbone(x)
        output = self.metric_fc(feature, label)

        return output



# --- ArcFace loss
# --- https://github.com/ronghuaiyang/arcface-pytorch
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            scale: norm of input feature
            margin: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, scale=30.0, margin=0.30, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):

        # when inference, label must be None
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
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.scale
        # print(output)

        return output


    def inference(self, input):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        return cosine


