import deform3d_double.deform_conv3d_op as deform_conv
import torch
import os
from deform3d_double.deform_conv3d_functions import ConvOffset3dFunction

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

batchsize = 2
c_in = 1
c_out = 1
inpu = 3
kernel = 1
stri = 1
pad = 0
out = int((inpu + 2 * pad - kernel) / stri + 1)
channel_per_group = 1
g_off = c_in // channel_per_group

inpu = torch.DoubleTensor([[[[[1.0, 1, 1], [2.0, 2, 2], [1.0, 1, 1]]] * inpu] * c_in] * batchsize).cuda()
offset = torch.DoubleTensor([[[[[0.000] * out] * out] * out,
                              [[[0.000] * out] * out] * out,
                              [[[0.0] * out] * out] * out]
                             * kernel * kernel * kernel * g_off] * batchsize).cuda()
weight = torch.DoubleTensor([[[[[1.0] * kernel] * kernel] * kernel] * c_in] * c_out).cuda()

padding = [pad, pad, pad]
stride = [stri, stri, stri]

conv_offset = ConvOffset3dFunction(stride, padding, channel_per_group)
output = conv_offset.forward(inpu, offset, weight)
# output_ = output.cpu().numpy()

grad_output = torch.DoubleTensor([[[[[1.0] * out] * out] * out] * c_out] * batchsize).cuda()
columns = torch.zeros(weight.size(1) * weight.size(2) * weight.size(3) * weight.size(4),
                      output.size(2) * output.size(3) * output.size(4)).double().cuda()

grad_input = inpu.new(inpu.size()).zero_()
grad_offset = offset.new(offset.size()).zero_()
deform_conv.deform_conv_backward_input_offset_cuda(
    inpu, weight, offset, grad_output, columns, grad_input, grad_offset,
    pad, pad, pad,
    stri, stri, stri, channel_per_group)
grad_input_ = grad_input.cpu().numpy()
grad_offset_ = grad_offset.cpu().numpy()

grad_weight = weight.new(weight.size()).zero_()
deform_conv.deform_conv_backward_weight_cuda(
    inpu, offset, grad_output, columns, grad_weight,
    pad, pad, pad,
    stri, stri, stri, channel_per_group)
grad_weight_ = grad_weight.cpu().numpy()

# print('forward\n', output_)
# print('grad input\n', grad_input_)
# print('grad weight\n', grad_weight_)
# print('grad offset\n', grad_offset_)
