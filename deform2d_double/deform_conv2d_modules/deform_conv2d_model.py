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
                 channel_per_group=1,
                 bias=True,
                 groups=1):
        super(ConvOffset2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.channel_per_group = channel_per_group
        self.group = groups

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.weight = nn.Parameter(torch.cuda.DoubleTensor(out_channels, in_channels // groups, *self.kernel_size))
        nn.init.kaiming_normal(self.weight.data, mode='fan_out')
        nn.init.constant(self.weight.data, 1)
        if bias:
            self.bias = nn.Parameter(torch.cuda.DoubleTensor(out_channels))
            nn.init.uniform(self.bias.data, -0.1, 0.1)
        else:
            self.register_parameter('bias', None)

    def forward(self, input, offset):
        # ConvOffset2dFunction.init(self.stride, self.padding, self.channel_per_group)
        return ConvOffset2dFunction.apply(input, offset, self.weight, self.bias, self.stride, self.padding,
                                          self.channel_per_group, self.group)
        # return ConvOffset2dFunction(self.stride, self.padding, self.channel_per_group)\
        #     .apply(input, offset, self.weight, self.bias)
