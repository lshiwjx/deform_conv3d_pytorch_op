import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from deform2d.deform_conv2d_modules import ConvOffset2d
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
batchsize = 200
c_in = 16
c_out = 16
inpu = 100
kernel = 3
stri = 1
pad = 0
out = int((inpu + 2 * pad - kernel) / stri + 1)
channel_per_group = 1

conv_offset2d = ConvOffset2d(c_in, c_out, kernel, stri, pad, channel_per_group)
conv = nn.Conv2d(
    c_in,
    c_in // channel_per_group * 2 * kernel * kernel,
    kernel_size=kernel,
    stride=stri,
    padding=pad,
    bias=False).type(torch.DoubleTensor).cuda()
inputs = Variable(torch.ones((batchsize, c_in, inpu, inpu)), requires_grad=True).type(torch.DoubleTensor).cuda()
# offset = Variable(torch.zeros((batchsize, c_in // channel_per_group * 2 * kernel * kernel, out, out)),
#                   requires_grad=True).type(torch.DoubleTensor).cuda()
offset = conv(inputs)
start = time.time()
output = conv_offset2d(inputs, offset)
forward = time.time() - start
print('time for forward: ', forward)

residual = Variable(torch.ones(output.size())).type(torch.DoubleTensor).cuda()
output.backward(residual)
print('backward', time.time() - forward - start)
print(output.size())
