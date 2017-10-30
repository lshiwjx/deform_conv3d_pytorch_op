import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from deform2d.deform_conv2d_modules import ConvOffset2d
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
batchsize = 2
c_in = 2
c_out = 2
inpu = 5
kernel = 3
stri = 1
pad = 1
out = int((inpu + 2 * pad - kernel) / stri + 1)
channel_per_group = 3

conv_offset2d = ConvOffset2d(c_in, c_out, kernel, stri, pad, channel_per_group)

inputs = Variable(torch.ones((batchsize, c_in, inpu, inpu)), requires_grad=True).cuda()
offset = Variable(torch.zeros((batchsize, c_in // channel_per_group * 2 * kernel * kernel, out, out)),
                  requires_grad=True).cuda()
start = time.time()
output = conv_offset2d(inputs, offset)
forward = time.time() - start
print('time for forward: ', forward)

residual = Variable(torch.ones(output.size())).cuda()
output.backward(residual)
print('backward', time.time() - forward - start)
print(output.size())
