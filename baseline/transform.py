from albumentations import *
from albumentations.pytorch import ToTensorV2
import ttach as tta

class BaseAugmentation:
    def __init__(self, train=True, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.mean = (0.548, 0.504, 0.479)
        self.std = (0.237, 0.247, 0.246)
        if train:
            self.transform = Compose([#CenterCrop(384,384),
                                    Resize(384,288),
                                    ShiftScaleRotate(rotate_limit=20,shift_limit=0.02, p=0.5), 
                                    HorizontalFlip(p=0.5),
                                    OneOf([Blur(blur_limit=3, p=1),
                                            MotionBlur(blur_limit=3, p=1)],p=0.5),
                                    RandomBrightnessContrast(0.1, 0.1,p=0.5),
                                    Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225]),
                                    ToTensorV2(),
                                    ])
        else:
            self.transform = Compose([#CenterCrop(384,384),
                            Resize(384,288),
                            Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225]),
                            ToTensorV2(),
                            ])
    def __call__(self, image):
        return self.transform(image=image)

class Augmentation_384:
    def __init__(self, train=True, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
        if train:
            self.transform = Compose([#CenterCrop(384,384),
                                    Resize(384,288),
                                    ShiftScaleRotate(rotate_limit=25,shift_limit=0.0325, scale_limit=[-0.0,0.2], p=0.5), 
                                    HorizontalFlip(p=0.5),
                                    
                                    OneOf([CLAHE(clip_limit=1.0, p=0.5),
                                        HueSaturationValue(p=0.5),
                                        RGBShift(p=0.5)],p=0.5),
                                    RandomBrightnessContrast(0.2, 0.3,p=0.5),
                                     OneOf(
                                         [
                                          Perspective(p=0.5),
                                     GridDistortion(p=0.5),
                                     OpticalDistortion(p=0.5),], p=0.2
                                     ),    
                                    Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225]),
                                    ToTensorV2(),
                                    ])
        else:
            self.transform = Compose([#CenterCrop(384,384),
                            Resize(384,288),
                            Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225]),
                            ToTensorV2(),
                            ])
    def __call__(self, image):
        return self.transform(image=image)
    
def get_tta_transform():
    transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        # tta.VerticalFlip(),
        # #밝기
        tta.Multiply(factors=[1.3, 1])
    ])

    return transforms