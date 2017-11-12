import torch
from deform3dl_double.deform_conv3dl_modules import ConvOffset3d
from torch.autograd import Variable
import os
from deform3dl_double.gradcheck import gradcheck
from deform3dl_double.deform_conv3dl_functions import ConvOffset3dFunction

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

batchsize = 2
c_in = 2
c_out = 3
inpu = 5
kernel = 3
stri = 2
pad = 1
out = int((inpu + 2 * pad - kernel) / stri + 1)
channel_per_group = 2
g_off = c_in // channel_per_group

conv_offset3d = ConvOffset3dFunction((stri, stri, stri), (pad, pad, pad), channel_per_group)

inputs = Variable(torch.rand(batchsize, c_in, inpu, inpu, inpu).type(torch.DoubleTensor).cuda(), requires_grad=True)
offsets = Variable(torch.rand(batchsize, g_off, out, out, out).type(torch.DoubleTensor).cuda(),
                   requires_grad=True)
weight = Variable(torch.rand(c_out, c_in, kernel, kernel, kernel).type(torch.DoubleTensor).cuda(),
                  requires_grad=True)

print(gradcheck(conv_offset3d, (inputs, offsets, weight)))
