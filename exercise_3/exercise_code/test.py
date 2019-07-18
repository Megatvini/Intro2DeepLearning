from torchvision import models
from torch import nn


def main():
    model = models.segmentation.fcn_resnet101(num_classes=23)

    print(model)


if __name__ == '__main__':
    main()
