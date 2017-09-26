import math

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _triple
from functions import conv_offset3d


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

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, offset):
        return conv_offset3d(input, offset, self.weight,
                             self.stride, self.padding, self.channel_per_group)
