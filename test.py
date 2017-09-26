import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import deform_conv3d_op as deform_conv

from modules import ConvOffset3d
from functions import ConvOffset3dFunction

# num_deformable_group = 1

# N, inC, inL, inH, inW = 1, 3, 3, 7, 7
# outC, outL, outH, outW = 4, 3,  7, 7
# kL, kH, kW = 3, 3, 3
# inpu = torch.FloatTensor([[[[[1., 1, 1]] * 3] * 3,[[[2., 2, 2]] * 3] * 3]*1]).cuda()

inpu = torch.FloatTensor([[[[[1., 1, 1]] * 3] * 3] * 1] * 1).cuda()
# inpu = torch.randn(1,1,3,7,7).cuda()
offset = torch.FloatTensor([[[[[0.0001] * 3] * 3] * 3, [[[0.0001] * 3] * 3] * 3, [[[3.0001] * 3] * 3] * 3] * 1]).cuda()
weight = torch.FloatTensor([[[[[1.0] * 1] * 1] * 1] * 1] * 1).cuda()

inputs = Variable(inpu).cuda()
offsets = Variable(offset).cuda()
weights = Variable(weight).cuda()

# conv = nn.Conv3d(
#     inC,
#     num_deformable_group * 3 * kL * kH * kW,
#     kernel_size=(kL, kH, kW),
#     stride=(1, 1, 1),
#     padding=(1, 1, 1),
#     bias=False).cuda()
# conv_offset3d = ConvOffset3d(inC, outC, (kL,kH, kW), stride=1, padding=1).cuda()

# inputs = Variable(torch.randn(N, inC, inL, inH, inW).cuda())
# offset = conv(inputs)

# output = conv_offset3d(inputs, offset)
# output.backward(output.data)
padding = [0, 0, 0]
stride = [1, 1, 1]
conv_offset = ConvOffset3dFunction(stride, padding)
output = conv_offset.forward(inpu, offset, weight)
tmp = output.cpu().numpy()

grad_output = torch.FloatTensor([[[[[1.0] * 3] * 3] * 3] * 1] * 1).cuda()
grad_input = inpu.new(*inpu.size()).zero_()
grad_offset = offset.new(*offset.size()).zero_()
bufs_ = [inpu.new(), inpu.new()]

deform_conv.deform_conv_backward_input_cuda(
    inpu, offset, grad_output, grad_input, grad_offset,
    weight, bufs_[0],
    weight.size(2), weight.size(3), weight.size(4), 1, 1, 1,
    0, 0, 0, 1, 1, 1, 1)
tmp3 = grad_input.cpu().numpy()
tmp4 = grad_offset.cpu().numpy()

grad_weight = weight.new(*weight.size()).zero_()
deform_conv.deform_conv_backward_parameters_cuda(
    inpu, offset, grad_output, grad_weight, bufs_[0],
    bufs_[1],
    weight.size(2), weight.size(3), weight.size(4), stride[0], stride[1], stride[2],
    padding[0], padding[1], padding[2], dilation[0], dilation[1],
    dilation[2], 1, 1)
tmp5 = grad_weight.cpu().numpy()

print(tmp, '\n', tmp3, '\n', tmp5, '\n', tmp4)
print(output.size())
