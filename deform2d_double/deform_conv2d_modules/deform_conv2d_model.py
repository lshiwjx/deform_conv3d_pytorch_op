import math

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair

from deform2d_double.deform_conv2d_functions import ConvOffset2dFunction


class ConvOffset2d(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 channel_per_group=1):
        super(ConvOffset2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.channel_per_group = channel_per_group

        self.weight = nn.Parameter(torch.cuda.DoubleTensor(out_channels, in_channels, *self.kernel_size))

        nn.init.kaiming_normal(self.weight.data, mode='fan_out')

    def forward(self, input, offset):
        return ConvOffset2dFunction(self.stride, self.padding, self.channel_per_group)(input, offset, self.weight)
