import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(-1, np.prod(x.size()[1:]))


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        ##############################################################################################################
        # TODO: Define all the layers of this CNN, the only requirements are:                                        #
        # 1. This network takes in a square (same width and height), grayscale image as input                        #
        # 2. It ends with a linear layer that represents the keypoints                                               #
        # it's suggested that you make this last layer output 30 values, 2 for each of the 15 keypoint (x, y) pairs  #
        #                                                                                                            #
        # Note that among the layers to add, consider including:                                                     #
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or      #
        # batch normalization) to avoid overfitting.                                                                 #
        ##############################################################################################################

        """
        Layer Number Layer Name Layer Shape
        1 Input1 (1, 96, 96)
        2 Convolution2d1 (32, 93, 93)
        3 Activation1 (32, 93, 93)
        4 Maxpooling2d1 (32, 46, 46)
        5 Dropout1 (32, 46, 46)
        6 Convolution2d2 (64, 44, 44)
        7 Activation2 (64, 44, 44)
        8 Maxpooling2d2 (64, 22, 22)
        9 Dropout2 (64, 22, 22)
        10 Convolution2d3 (128, 21, 21)
        11 Activation3 (128, 21, 21)
        12 Maxpooling2d3 (128, 10, 10)
        13 Dropout3 (128, 10, 10)
        14 Convolution2d4 (256, 10, 10)
        15 Activation4 (256, 10, 10)
        16 Maxpooling2d4 (256, 5, 5)
        17 Dropout4 (256, 5, 5)
        18 Flatten1 (6400)
        19 Dense1 (1000)
        20 Activation5 (1000)
        21 Dropout5 (1000)
        22 Dense2 (1000)
        23 Activation6 (1000)
        24 Dropout6 (1000)
        25 Dense3 (2)
        """
        ########################################################################
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=2),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=1),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4),
            nn.BatchNorm2d(256),

            Flatten(),

            nn.Linear(6400, 1000),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(1000),

            nn.Linear(1000, 30)
        )
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        ##################################################################################################
        # TODO: Define the feedforward behavior of this model                                            #
        # x is the input image and, as an example, here you may choose to include a pool/conv step:      #
        # x = self.pool(F.relu(self.conv1(x)))                                                           #
        # a modified x, having gone through all the layers of your model, should be returned             #
        ##################################################################################################
        x = self.model(x)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
