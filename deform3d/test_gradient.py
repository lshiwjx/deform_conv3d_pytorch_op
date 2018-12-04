import torch
from deform3d.deform_conv3d_functions import ConvOffset3dFunction
from torch.autograd import Variable
import os
from deform3d.gradcheck import gradcheck
import torch.nn as nn
# from torch.autograd import gradcheck

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
batchsize = 2
c_in = 2
c_out = 4
inpu = 7
kernel = 3
stri = 2
pad = 2
dilation = 2
out = int((inpu + 2 * pad - kernel) / stri + 1)
channel_per_group = 2
g_off = c_in // channel_per_group
c_off = g_off * kernel * kernel * kernel * 3
group = 2

inputs = Variable(torch.rand(batchsize, c_in, inpu, inpu, inpu).cuda(), requires_grad=True)
offsets = Variable(torch.rand(batchsize, c_off, out, out, out).cuda(), requires_grad=True)
weight = Variable(torch.rand(c_out, c_in // group, kernel, kernel, kernel).cuda(), requires_grad=True)
bias = Variable(torch.rand(c_out).cuda(), requires_grad=True)

print(gradcheck(ConvOffset3dFunction.apply,
                (inputs, offsets, weight, bias, (stri, stri, stri), (pad, pad, pad), (dilation, dilation, dilation),
                 channel_per_group, group)))
# print(gradcheck(F.conv3d, (inputs, weight)))
