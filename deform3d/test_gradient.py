import torch
from deform3d.deform_conv3d_functions import ConvOffset3dFunction
from torch.autograd import Variable
import os
from deform3d.gradcheck import gradcheck
import torch.nn as nn
# from torch.autograd import gradcheck
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
batchsize = 1
c_in = 1
c_out = 1
inpu = 3
kernel = 1
stri = 1
pad = 0
out = int((inpu + 2 * pad - kernel) / stri + 1)
channel_per_group = 1
g_off = c_in // channel_per_group
c_off = g_off * kernel * kernel * kernel * 3

conv_offset3d = ConvOffset3dFunction((stri, stri, stri), (pad, pad, pad), channel_per_group)
inputs = Variable(torch.rand(batchsize, c_in, inpu, inpu, inpu).cuda(), requires_grad=True)
offsets = Variable(torch.rand(batchsize, c_off, out, out, out).cuda(), requires_grad=True)
weight = Variable(torch.rand(c_out, c_in, kernel, kernel, kernel).cuda(), requires_grad=True)

# print(gradcheck(conv_offset3d, (inputs, offsets, weight)))
print(gradcheck(F.conv3d, (inputs, weight)))
