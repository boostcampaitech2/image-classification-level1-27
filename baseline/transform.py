from albumentations import *
from albumentations.pytorch import ToTensorV2

class BaseAugmentation:
    def __init__(self, train=True, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.mean = (0.548, 0.504, 0.479)
        self.std = (0.237, 0.247, 0.246)
        if train:
            self.transform = Compose([#CenterCrop(384,384),
                                    Resize(512,384),
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
                            Resize(512,384),
                            Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225]),
                            ToTensorV2(),
                            ])
    def __call__(self, image):
        return self.transform(image=image)
