
import sys
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn

import functools
import operator

import math
from torchsummary import summary
from torchvision import transforms
from torch.nn.init import xavier_normal_
from torch.nn import functional as F
from tqdm.notebook import tqdm
from glob import glob
from numpy.random import randint
from turtle import forward
from torch.nn import Conv2d, Sequential
from config import ModelAndTrainingConfig as config


class IdentiyBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(IdentiyBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=1, padding="same", bias=bias)
        self.instance_norm_1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding="same", bias=bias)
        self.instance_norm_2 = nn.InstanceNorm2d(out_channels)

    def forward(self, x_in):
        input = x_in
        x = self.conv1(x_in)
        x = self.instance_norm_1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.instance_norm_2(x)
        return x + input


def seg_down_sample(seg, scale_factor_h, scale_factor_w):
    _, _, h, w = seg.shape
    new_size = [h // scale_factor_h, w // scale_factor_w]
    resized_img = transforms.functional.resize(seg, new_size)
    return resized_img


def param_free_norm(x, epsilon=1e-5):
    x_mean, x_var = torch.mean(x, dim=(2, 3)), torch.var(x, dim=(2, 3))
    x_std = torch.sqrt(x_var + epsilon)
    return (x - x_mean[..., None, None]) * (1 / x_std)[..., None, None]


class Spade(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        """
        Input and output of this layer is always of the same shape
        """
        super(Spade, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        self.net = nn.Sequential(
            nn.Conv2d(config.c_dim, out_channels=128, kernel_size=5, stride=1, padding=2, bias=self.bias),
            nn.ReLU(inplace=False)
        )

        self.gamma = nn.Conv2d(128, self.out_channels, kernel_size=5, stride=1, padding=2, bias=self.bias)
        self.beta = nn.Conv2d(128, self.out_channels, kernel_size=5, stride=1, padding=2, bias=self.bias)

    def forward(self, segmap, x_in):
        x = param_free_norm(x_in)
        _, _, x_h, x_w, = x_in.shape
        segmap_down = transforms.functional.resize(segmap, (x_h, x_w))
        segmap_down = self.net(segmap_down)

        segmap_gamma = self.gamma(segmap_down)
        segmap_beta = self.beta(segmap_down)

        x = x * (1 + segmap_gamma) + segmap_beta
        return x


class SpadeResblock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(SpadeResblock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        channel_middle = min(in_channels, out_channels)
        self.spade1 = Spade(in_channels, in_channels, bias=bias)
        self.conv1 = nn.Conv2d(in_channels=channel_middle,
                               out_channels=channel_middle,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=bias)
        self.leakyRelu1 = nn.LeakyReLU(0.2, inplace=False)

        self.spade2 = Spade(in_channels=channel_middle,
                            out_channels=channel_middle,
                            bias=bias)
        self.leakyRelu2 = nn.LeakyReLU(0.2, inplace=False)

        self.conv2 = nn.Conv2d(in_channels=channel_middle,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=bias)

        self.extra_spade = Spade(in_channels=in_channels,
                                 out_channels=in_channels,
                                 bias=bias)
        self.extra_cov = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1, stride=1,
                                   bias=False)

    def forward(self, segmap, x_init):
        x = self.spade1(segmap, x_init)
        x = self.leakyRelu1(x)
        x = self.conv1(x)
        x = self.spade2(segmap, x)
        x = self.leakyRelu1(x)
        x = self.conv2(x)

        if self.in_channels != self.out_channels:
            x_init = self.extra_spade(segmap, x_init)
            x_init = self.extra_cov(x_init)

        return x + x_init


class Conv2dSamePadding(nn.Conv2d):
    """
    This conv layer is used if we want to make the shape
    of convolution more predictable.
    """

    def __init__(self, *args, **kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)

    def get_padding_for_same(self, kernel_size, stride, padding, input: torch.Tensor):
        if isinstance(padding, int):
            input = F.pad(input, (padding, padding, padding, padding))
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        _, _, H, W = input.shape
        s_H = stride[0]
        s_W = stride[1]
        k_H = kernel_size[0]
        k_W = kernel_size[1]
        h2 = math.floor(H / s_H)
        w2 = math.floor(W / s_W)
        pad_W = (w2 - 1) * s_W + (k_W - 1) + 1 - W
        pad_H = (h2 - 1) * s_H + (k_H - 1) + 1 - H
        padding = (pad_W // 2, pad_W - pad_W // 2, pad_H // 2, pad_H - pad_H // 2)
        return padding

    def forward(self, input):
        padding = self.get_padding_for_same(self.kernel_size, self.stride, self.padding, input)
        return self._conv_forward(F.pad(input, padding), self.weight, self.bias)


class AdaptiveNoiseMultiplier(nn.Module):
    def __init__(self):
        super(AdaptiveNoiseMultiplier, self).__init__()
        self.net = Sequential(
            Conv2dSamePadding(config.noise_dim[0], 32, 5, 2),  # spotial // 2
            Conv2dSamePadding(32, 64, 3, 2),  # spatial // 2
            nn.ReLU(inplace=False),
            nn.Flatten(),
            nn.Linear(64 * np.prod(config.noise_dim[1:3]) // (4**2), 128),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, z_in):
        return self.net(z_in)


if __name__ == "__main__":
    result = Conv2dSamePadding(3, 6, 100, 3)(torch.randn(10, 3, 300, 300))
    print(result.shape)
    result = Conv2dSamePadding(3, 6, 100, 3)(torch.randn(10, 3, 299, 299))
    print(result.shape)
    result = Conv2dSamePadding(3, 6, 100, 3)(torch.randn(10, 3, 298, 298))
    print(result.shape)
    result = Conv2dSamePadding(3, 6, 100, 3)(torch.randn(10, 3, 297, 297))
    print(result.shape)
