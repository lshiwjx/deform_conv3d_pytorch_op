import torch
from deform_conv2d_modules import ConvOffset2d
from deform_conv2d_functions import ConvOffset2dFunction
from torch.autograd import Variable
import os
from gradcheck import gradcheck
import torch.nn.functional as F
# from deform2d.modules import ConvOffset2d
# from deform2d.functions import conv_offset2d

batchsize = 2
c_in = 2
c_out = 3
inpu = 7
kernel = 1
stri = 2
pad = 1
out = int((inpu + 2 * pad - kernel) / stri + 1)
channel_per_group = 2
g_off = c_in // channel_per_group
c_off = g_off * kernel * kernel * 2

# s = Variable(torch.rand(batchsize,2,3).type(torch.FloatTensor).cuda(), requires_grad=True)
# x = Variable(torch.rand(batchsize, c_in, inpu, inpu).type(torch.FloatTensor).cuda(), requires_grad=True)
# def stn(s, x):
#     g = F.affine_grid(s, x.size())
#     x = F.grid_sample(x, g)
#     return x
# print(gradcheck(stn, (s, x)))


# conv_offset2d = ConvOffset2d(c_in, c_out, kernel, stri, pad, channel_per_group).cuda()
conv_offset2d = ConvOffset2dFunction((stri, stri), (pad, pad), channel_per_group)
# conv_offset2d = ConvOffset2d(c_in, c_out, kernel, stri, pad, channel_per_group).cuda()

# inputs = Variable(torch.FloatTensor([[[[1., 1., 1, 1, 1]] * 5] * c_in] * batchsize).type(torch.FloatTensor).cuda(),
#                   requires_grad=True)
# offsets = Variable(torch.FloatTensor([[[[0.001] * out] * out,
#                                        [[0.001] * out] * out]
#                                       * kernel * kernel * g_off] * batchsize).type(torch.FloatTensor).cuda(),
#                    requires_grad=False)
# weight = Variable(torch.ones(c_out, c_in, kernel, kernel).type(torch.FloatTensor).cuda(),
#                   requires_grad=False)
#
inputs = Variable(torch.rand(batchsize, c_in, inpu, inpu).type(torch.FloatTensor).cuda(), requires_grad=True)
offsets = Variable(torch.rand(batchsize, c_off, out, out).type(torch.FloatTensor).cuda(),
                   requires_grad=True)
weight = Variable(torch.rand(c_out, c_in, kernel, kernel).type(torch.FloatTensor).cuda(),
                  requires_grad=True)

print(gradcheck(conv_offset2d, (inputs, offsets, weight)))
