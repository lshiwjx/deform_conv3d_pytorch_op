import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from conv2d.conv2d_modules import Conv2d
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
batchsize = 2
c_in = 4
c_out = 8
inpu = 75
kernel = 3
stri = 1
pad = 1
dila = 3
out = int((inpu + 2 * pad - dila * (kernel - 1) - 1) / stri + 1)
group = 4

conv1 = Conv2d(c_in, c_out, kernel, stri, pad, dila, bias=True, groups=group).type(torch.cuda.FloatTensor)
torch.manual_seed(1)
nn.init.uniform(conv1.weight.data, -1, 1)
nn.init.uniform(conv1.bias.data, -1, 1)
conv2 = nn.Conv2d(c_in, c_out, kernel, stri, pad, dila, bias=True, groups=group).type(torch.cuda.FloatTensor)
torch.manual_seed(1)
nn.init.uniform(conv2.weight.data, -1, 1)
nn.init.uniform(conv2.bias.data, -1, 1)

inputs = Variable(torch.rand((batchsize, c_in, inpu, inpu)), requires_grad=True).type(torch.cuda.FloatTensor)

start = time.clock()
output1 = conv1(inputs)
end1 = time.clock()
output2 = conv2(inputs)
end2 = time.clock()
forward1 = end1 - start
forward2 = end2 - end1
print('time for forward: ', forward1, ' ', forward2)

residual = Variable(torch.ones(output1.size())).type(torch.FloatTensor).cuda()
start = time.clock()
output1.backward(residual)
end1 = time.clock()
output2.backward(residual)
end2 = time.clock()
backward1 = end1 - start
backward2 = end2 - start
print('time for backward: ', backward1, ' ', backward2)

print('finish')
