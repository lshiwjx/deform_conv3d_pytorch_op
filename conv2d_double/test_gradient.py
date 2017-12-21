import torch
from conv2d_double.conv2d_functions import Conv2dFunction
from torch.autograd import Variable
import os
from conv2d_double.gradcheck import gradcheck

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
batchsize = 2
c_in = 2
c_out = 4
inpu = 7
kernel = 1
stri = 2
pad = 2
dilation = 2
out = int((inpu + 2 * pad - dilation * (kernel - 1) - 1) / stri + 1)
group = 2

inputs = Variable(torch.rand(batchsize, c_in, inpu, inpu).double().cuda(), requires_grad=True)
weight = Variable(torch.rand(c_out, c_in // group, kernel, kernel).double().cuda(),
                  requires_grad=True)
bias = Variable(torch.rand(c_out).double().cuda(),
                requires_grad=True)

print(
    gradcheck(Conv2dFunction.apply,
              (inputs, weight, bias, (stri, stri), (pad, pad), (dilation, dilation), group)))
