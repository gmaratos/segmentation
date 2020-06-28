import torch
import torch.nn as nn
import torchvision
from .utils import SimpleSegmentation

from torch.nn.functional import adaptive_avg_pool2d as AdaptiveAvgPool

__all__ = ['VGG16_FCN']

class VGG16_FCN(SimpleSegmentation):
    """implements the vgg16_fcn model, using a pretrained backbone and an fcn head
    that is similar in design to the reference code in pytorch"""
    def __init__(self, num_classes: int = 21): #assumes voc if num_classes not given
        super().__init__(num_classes)
        self.backbone = VGG16_Backbone()
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )

class VGG16_Backbone(nn.Module):
    """pretrained on imagenet, vgg16 backbone, outputs 512 features."""
    def __init__(self):
        super().__init__()
        vgg16 = torchvision.models.vgg16_bn(pretrained=True)
        self.features = vgg16.features

    def forward(self, x):
        x = self.features(x)
        x = AdaptiveAvgPool(x, (7, 7))
        return x
