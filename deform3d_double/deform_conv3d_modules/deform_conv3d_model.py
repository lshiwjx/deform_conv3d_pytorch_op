import math

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _triple

from deform3d_double.deform_conv3d_functions import ConvOffset3dFunction


class ConvOffset3d(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 channel_per_group=1):
        super(ConvOffset3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.channel_per_group = channel_per_group

        self.weight = nn.Parameter(torch.cuda.DoubleTensor(out_channels, in_channels, *self.kernel_size))
        nn.init.kaiming_normal(self.weight.data, mode='fan_out')

    def forward(self, input, offset):
        return ConvOffset3dFunction(self.stride, self.padding, self.channel_per_group)(input, offset, self.weight)
