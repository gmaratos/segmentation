import torch.nn as nn
from torch.nn.functional import interpolate as upsample_

class SimpleSegmentation(nn.Module):
    """combines the backbone and classifier and does an upsample so that we get a dense
    prediction"""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        #these are defined by classes that inherit this
        self.backbone = None
        self.classifier = None

    def forward(self, x):
        input_shape = x.shape[-2:] #the last two dimensios are assumed to be h an w
        x = self.backbone(x)
        x = self.classifier(x)
        x = upsample_(x, size=input_shape, mode='bilinear', align_corners=False)
        return {'out': x}
