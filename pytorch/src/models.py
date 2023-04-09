import numpy as np
import torch
import torch.nn as nn
import config
import layers
import copy

from numpy.random import randint
from glob import glob
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from torch.nn import Sequential, LeakyReLU, Conv2d, Flatten, Linear, ReLU, InstanceNorm2d
from torchvision import transforms
from layers import Conv2dSamePadding, IdentiyBlock, SpadeResblock, AdaptiveNoiseMultiplier
from torchsummary import summary
from config import ModelAndTrainingConfig as config


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        channel = config.init_channels

        self.conv1 = Sequential(
            Conv2dSamePadding(config.image_shape[0], channel, kernel_size=4, stride=2, bias=True),
            ReLU(inplace=False)
        )
        self.block = self._block()
        self.critic = Sequential(
            Conv2dSamePadding(2048, 1, kernel_size=3, stride=1, bias=False),
            Flatten(),
            Linear(9, 1)
        )
        c_kernel = int(config.image_shape[1] / np.power(2, config.n_dis))
        self.cls_logit = Sequential(
            Conv2d(2048, config.c_dim, kernel_size=c_kernel, stride=1, bias=False),
            Flatten()
        )
        self.cls_relu = nn.ReLU(inplace=False)

    def _block(self):
        channel = copy.deepcopy(config.init_channels)
        layers = []
        for i in range(1, config.n_dis):
            layers.append(Conv2dSamePadding(channel, channel * 2, kernel_size=4, stride=2, bias=True))
            layers.append(LeakyReLU(0.01, inplace=False))
            channel = channel * 2

        return Sequential(*layers)

    def forward(self, x_in):
        x = self.conv1(x_in)
        x = self.block(x)

        critic = self.critic(x)

        cls_logit = self.cls_logit(x)
        cls_logit = self.cls_relu(cls_logit)

        return critic, cls_logit


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        channel = copy.deepcopy(config.init_channels)
        self.adaptive_noise_mul = AdaptiveNoiseMultiplier()
        #                      6
        self.conv1 = Conv2d(3 + config.noise_dim[0], channel, kernel_size=7, stride=1, padding=3, bias=False)
        self.instance_norm = nn.InstanceNorm2d(channel)
        self.relu = ReLU(inplace=False)
        self.down_sample = self._down_sample()
        self.bottle_neck = self._bottleneck()

        channel = channel * (2**3)
        self.up_conv2d_1 = nn.ConvTranspose2d(channel, channel // 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.up_spade_blk_1 = SpadeResblock(channel // 2, channel // 2)
        self.up_relu_1 = ReLU(inplace=False)

        self.up_conv2d_2 = nn.ConvTranspose2d(channel // 2, channel // 4, kernel_size=4, stride=2, padding=1, bias=False)
        self.up_spade_blk_2 = SpadeResblock(channel // 4, channel // 4)
        self.up_relu_2 = ReLU(inplace=False)

        self.up_conv2d_3 = nn.ConvTranspose2d(channel // 4, channel // 8, kernel_size=4, stride=2, padding=1, bias=False)
        self.up_spade_blk_3 = SpadeResblock(channel // 8, channel // 8)
        self.up_relu_3 = ReLU(inplace=False)

        self.mask = Sequential(
            nn.Conv2d(config.init_channels, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.overlay = Sequential(
            Conv2d(config.init_channels, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()
        )

    def _down_sample(self):
        layers = []
        channel = copy.deepcopy(config.init_channels)
        for i in range(3):
            layers.append(
                Conv2dSamePadding(channel, channel * 2, kernel_size=4, stride=2, bias=False)
            )
            layers.append(InstanceNorm2d(channel * 2))
            layers.append(ReLU(inplace=False))
            channel = channel * 2
        return Sequential(*layers)

    def _bottleneck(self):
        channel = config.init_channels
        layers = []
        for i in range(config.n_res):
            layers.append(IdentiyBlock(in_channels=channel * (2**3), out_channels=channel * (2**3), bias=False))
        return Sequential(*layers)

    def forward(self, x_in, c_in, z):
        """
        Args:
            x_in (Tensor): input image, config.img_shape
            c_in (Tensor): input categorical and spatial map (shape: (*config.img_shape[1:3], c_dim))
            z (Tensor):    noise of dim (-, config.noise_dim)
        """
        lambda_z = self.adaptive_noise_mul(z)
        x = torch.cat((x_in, lambda_z[:, :, None, None] * z), dim=1)
        x = self.conv1(x)
        x = self.instance_norm(x)
        x = self.relu(x)

        x = self.down_sample(x)

        x = self.bottle_neck(x)

        x = self.up_conv2d_1(x)
        x = self.up_spade_blk_1(c_in, x)
        x = self.up_relu_1(x)

        x = self.up_conv2d_2(x)
        x = self.up_spade_blk_2(c_in, x)
        x = self.up_relu_2(x)

        x = self.up_conv2d_3(x)
        x = self.up_spade_blk_3(c_in, x)
        x = self.up_relu_3(x)

        mask = self.mask(x)
        defect_overlay = self.overlay(x)

        return defect_overlay, mask


if __name__ == "__main__":
    gen = Generator()
    x = gen(torch.randn(1, 3, 224, 224), torch.randn(1, 2, 224, 224), torch.randn(1, 3, 224, 224))
    print(x)
