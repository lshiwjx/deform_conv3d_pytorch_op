import torch
from deform2d.deform_conv2d_modules import ConvOffset2d
from deform2d.deform_conv2d_functions import ConvOffset2dFunction
from torch.autograd import Variable
import os
from deform2d.gradcheck import gradcheck

# from deform2d.modules import ConvOffset2d
# from deform2d.functions import conv_offset2d

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
batchsize = 2
c_in = 1
c_out = 1
inpu = 5
kernel = 3
stri = 1
pad = 0
out = int((inpu + 2 * pad - kernel) / stri + 1)
channel_per_group = 1
g_off = c_in // channel_per_group
c_off = g_off * kernel * kernel * 2

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

inputs = Variable(torch.rand(batchsize, c_in, inpu, inpu).type(torch.FloatTensor).cuda(), requires_grad=True)
offsets = Variable(torch.rand(batchsize, c_off, out, out).type(torch.FloatTensor).cuda(),
                   requires_grad=True)
weight = Variable(torch.rand(c_out, c_in, kernel, kernel).type(torch.FloatTensor).cuda(),
                  requires_grad=True)

print(gradcheck(conv_offset2d, (inputs, offsets, weight)))
