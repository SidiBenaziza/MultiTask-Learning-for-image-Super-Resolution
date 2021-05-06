import torch
from torch.utils.data import Dataset,DataLoader
from torch import nn
from PIL import Image, ImageOps
from torchsummary import summary
import torch.nn.functional as F

import numpy as np
from torch.utils.data import Subset
import os
from tqdm import tqdm
import cv2 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils import AverageMeter, calc_psnr
import copy


from functools import partial
from collections import OrderedDict


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size
 

def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels
 
class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, stride=1, conv=partial(Conv2dAuto, kernel_size=3, bias=False) , *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.stride, self.conv = expansion, stride, conv
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.stride, bias=False) ) if self.should_apply_shortcut else None

           # nn.BatchNorm2d(self.expanded_channels)) 
        
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs))# nn.BatchNorm2d(out_channels)


class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.stride),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )



class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        stride = 1
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, stride=stride),
            *[block(out_channels * block.expansion, 
                    out_channels, stride=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetSR(nn.Module):
    def __init__(self, in_channels=1, blocks_sizes=[64,64,64,64], deepths=[3,3,3,3], 
                 activation='relu', block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes
        
        self.gate = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, padding=3 // 2))
        # nn.Upsample(scale_factor=2,mode="bicubic")
         # nn.BatchNorm2d(in_channels) 
          

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            
            *[ResNetLayer(in_channels * block.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]       
        ])
        
        self.hr_out = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=3 // 2),
            nn.PixelShuffle(upscale_factor=2),
            activation_func(activation), 
            nn.Conv2d(16,1, kernel_size=3, padding=3 // 2)
            )

        self.hq_out = nn.Conv2d(64, 1, kernel_size=3, padding=3 // 2)
        
    def forward(self, x):
        x = self.gate(x)
        
        for block in self.blocks:
            x = block(x)
        
        hr = self.hr_out(x)

        hq = self.hq_out(x)

        return hr, hq
   