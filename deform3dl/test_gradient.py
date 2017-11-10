import torch
from deform3dl.deform_conv3dl_modules import ConvOffset3d
from torch.autograd import Variable
import os
from deform3dl.gradcheck import gradcheck
from deform3dl.deform_conv3dl_functions import ConvOffset3dFunction
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

batchsize = 2
c_in = 2
c_out = 3
in_l = in_h = in_w = 5
kernel_l = kernel_h = kernel_w = 3
out_l = out_h = out_w = 3
stri = 1
pad = 0
channel_per_group = 1
c_off = c_in // channel_per_group

conv_offset3d = ConvOffset3dFunction((stri, stri, stri), (pad, pad, pad), channel_per_group)

inputs = Variable(torch.rand(batchsize, c_in, in_l, in_h, in_w).cuda(), requires_grad=True)
offsets = Variable(torch.rand(batchsize, c_off, out_l, out_h, out_w).cuda(),
                   requires_grad=True)
weight = Variable(torch.rand(c_out, c_in, kernel_l, kernel_h, kernel_w).cuda(),
                  requires_grad=True)

print(gradcheck(conv_offset3d, (inputs, offsets, weight)))
