import deform3dl.deform_conv3dl_op as deform_conv
import torch
import os
from deform3dl.deform_conv3dl_functions import ConvOffset3dFunction

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

batchsize = 1
c_in = 1
c_out = 1
inpu = 3
kernel = 1
stri = 1
pad = 0
out = int((inpu + 2 * pad - kernel) / stri + 1)
channel_per_group = 1
g_off = c_in // channel_per_group

inpu = torch.FloatTensor(
    [[[[[1.0] * inpu] * inpu, [[2.0] * inpu] * inpu, [[1.0] * inpu] * inpu]] * c_in] * batchsize).cuda()
# inpu = torch.randn(1,1,3,7,7).cuda()
offset = torch.FloatTensor([[[[[1.] * out] * out] * out] * g_off] * batchsize).cuda()
weight = torch.FloatTensor([[[[[1.0] * kernel] * kernel] * kernel] * c_in] * c_out).cuda()
tmp = offset.cpu().numpy()

padding = [pad, pad, pad]
stride = [stri, stri, stri]
conv_offset = ConvOffset3dFunction(stride, padding, channel_per_group)
output = conv_offset.forward(inpu, offset, weight)
tmp = output.cpu().numpy()

grad_output = torch.FloatTensor([[[[[1.0] * out] * out] * out] * c_out] * batchsize).cuda()
columns = torch.zeros(weight.size(1) * weight.size(2) * weight.size(3) * weight.size(4),
                      output.size(2) * output.size(3) * output.size(4)).cuda()

# test for graph and grad
# inputs = Variable(torch.randn(batchsize,c_in,in,in,in)
#                   .cuda(),requires_grad=True)
# offsets = Variable(torch.randn(batchsize,g_c,out,out,out)
#                    .cuda(),requires_grad=True)
# weights = Variable(torch.randn(c_out,c_in,kernel,kernel,kernel)
#                    .cuda(), requires_grad=True)
# o1 = conv_offset(inputs, offsets, weights)
# o2 = Variable(grad_output,requires_grad=False)
# loss = (o1 - o2).sum()
# loss.backward()
# grad_i = inputs.grad.data.cpu().numpy()
# grad_o = offsets.grad.data.cpu().numpy()
# grad = weights.grad.data.cpu().numpy()
# ---------------------------------------------

grad_input = inpu.new(*inpu.size()).zero_()
grad_offset = offset.new(*offset.size()).zero_()
deform_conv.deform_conv_backward_input_offset_cuda(
    inpu, weight, offset, grad_output, columns, grad_input, grad_offset,
    pad, pad, pad,
    stri, stri, stri, channel_per_group)
grad_input_ = grad_input.cpu().numpy()
grad_offset_ = grad_offset.cpu().numpy()

gradeight = weight.new(*weight.size()).zero_()
deform_conv.deform_conv_backward_weight_cuda(
    inpu, offset, grad_output, columns, gradeight,
    pad, pad, pad,
    stri, stri, stri, channel_per_group)
gradeight_ = gradeight.cpu().numpy()

print('forward\n', tmp)
print('grad input\n', grad_input_)
print('grad weight\n', gradeight_)
print('grad offset\n', grad_offset_)
