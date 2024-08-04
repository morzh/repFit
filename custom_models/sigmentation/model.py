import os
import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchsummary import summary


class conv_1d_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size=7, stride=1, padding=0, activation=nn.Tanh):
        super().__init__()
        self.conv = nn.Conv1d(in_layer, out_layer, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_layer)
        self.activation = activation()

    def forward(self, x):
        x_re = self.conv(x)
        x_re = self.bn(x_re)
        x_out = self.activation(x_re)
        return x_out


class conv_2d_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size=7, stride=1, padding=0, activation=nn.Tanh):
        super().__init__()
        self.conv = nn.Conv2d(in_layer, out_layer, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_layer)
        self.activation = activation()

    def forward(self, x):
        x_re = self.conv(x)
        x_re = self.bn(x_re)
        x_out = self.activation(x_re)
        return x_out


class SegmentationModel(nn.Module):
    """ Model for exercise segmentation. Used sequence with length 200 frames. """

    def __init__(self):
        super().__init__()
        self.conv1 = conv_2d_block(1, 256, kernel_size=(52, 5), stride=1, padding=(0, 2))
        self.conv2 = conv_1d_block(257, 512, kernel_size=5, stride=2, padding=2)
        self.conv3 = conv_1d_block(512, 256, kernel_size=5, stride=2, padding=2)
        self.conv4 = conv_1d_block(256, 256, kernel_size=5, stride=5, padding=0)
        self.conv5 = conv_1d_block(256, 256, kernel_size=10, stride=1, padding=0)

        self.conv6 = conv_1d_block(256, 128, kernel_size=4, stride=1, padding=1)
        self.conv7 = conv_1d_block(640, 64, kernel_size=3, stride=1, padding=1)
        self.conv8 = conv_1d_block(321, 64, kernel_size=3, stride=1, padding=1)
        self.conv9 = conv_1d_block(65, 1, kernel_size=3, stride=1, padding=1)

        self.upsample2 = self.upsample(2)

    def upsample(self, scale_factor: float):
        def _upsample(x):
            return F.interpolate(x[None], scale_factor=(1, scale_factor), mode="bicubic").squeeze()
        return _upsample

    def forward(self, input):
        pca = input[:, 0, :].unsqueeze(1)
        # joints = input[:, 1:, :]
        x = input.unsqueeze(1)

        # downsample
        x1 = self.conv1(x).squeeze()
        x1 = torch.concatenate((x1, pca), dim=1)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        
        # upsample
        x6 = torch.concatenate((x5, x3), dim=2) 
        x7 = self.conv6(x6)

        x8 = self.upsample2(x7)
        x9 = torch.concatenate((x8, x2), dim=1)
        x10 = self.conv7(x9)

        x11 = self.upsample2(x10)
        x12 = torch.concatenate((x11, x1), dim=1)
        x13 = self.conv8(x12)

        x14 = torch.concatenate((x13, pca), dim=1)
        out = self.conv9(x14)
        return out

