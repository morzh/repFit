import os
import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchsummary import summary


class conv_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size=7, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv1d(in_layer, out_layer, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_layer)
        self.relu = nn.Tanh()

    def forward(self, x):
        x_re = self.conv(x)
        x_re = self.bn(x_re)
        x_out = self.relu(x_re)
        return x_out


class ModelClassifier(nn.Module):
    """  """

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim
        self.layers = 50
        self.conv1 = conv_block(1, self.layers, kernel_size=7, stride=1, padding=3)
        self.conv2 = conv_block(1, self.layers, kernel_size=7, stride=2, padding=3)

        self.upsample2 = self.upsample(2)

    def upsample(self, scale_factor: float):
        def _upsample(x):
            return F.interpolate(x[None], scale_factor=(1, scale_factor), mode="bicubic").squeeze()
        return _upsample

    def forward(self, input):
        x = input.reshape(input.shape[0], 1, input.shape[-1])

        x1 = self.conv1(x)

        return x1

