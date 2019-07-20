"""SegmentationNN"""
import torch
import torch.nn as nn
from torchvision import models


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def out(x):
    return x['out']


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################

        resnet = models.segmentation.fcn_resnet101(num_classes=num_classes, pretrained=False)
        # for p in resnet.parameters():
        #     p.required_grad = False

        self.model = nn.Sequential(
            resnet,
            Lambda(out),
            nn.Conv2d(21, 128, 3, padding=1),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.Conv2d(128, num_classes, 3, padding=1)
        )
        #
        # self.model = models.segmentation.fcn_resnet50(num_classes=num_classes)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        return self.model(x)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
