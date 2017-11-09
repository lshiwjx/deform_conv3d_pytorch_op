import torch
from deform3dl.deform_conv3dl_modules import ConvOffset3d
from torch.autograd import Variable
import os
from deform3dl.gradcheck import gradcheck

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

conv_offset3d = ConvOffset3d(c_in, c_out, kernel_l, stri, pad, channel_per_group).cuda()

# inpu = torch.FloatTensor([[[[[1., 2., 3, 1, 1]] * 5] * 5] * 1] * 1).cuda()
# off = torch.FloatTensor([[[[[0.5001]*3] * 3] * 3] * 81] * 1).cuda()
# conv_offset3d = ConvOffset3d(1, 1, (3, 3, 3), stride=1, padding=0).cuda()

inputs = Variable(torch.rand(batchsize, c_in, in_l, in_h, in_w).type(torch.FloatTensor).cuda(), requires_grad=True)
offsets = Variable(torch.rand(batchsize, c_off, out_l, out_h, out_w).type(torch.FloatTensor).cuda(),
                   requires_grad=True)
inpu = inputs.data.cpu().numpy()
off = offsets.data.cpu().numpy()

print(gradcheck(conv_offset3d, (inputs, offsets), atol=0.1))
