import math

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair

from conv2d_double.conv2d_functions import Conv2dFunction


class Conv2d(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 groups=1):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
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

    def forward(self, inputs):
        return Conv2dFunction.apply(inputs, self.weight, self.bias, self.stride, self.padding, self.dilation,
                                    self.group)
