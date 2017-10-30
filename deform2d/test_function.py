import deform2d.deform_conv2d_op as deform_conv
import deform2d._ext.deform_conv as zb
import torch
import os
from deform2d.deform_conv2d_functions import ConvOffset2dFunction as my
from deform2d.functions.deform_conv import ConvOffset2dFunction

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
batchsize = 2
c_in = 2
c_out = 3
inpu = 5
kernel = 3
stri = 1
pad = 0
out = int((inpu + 2 * pad - kernel) / stri + 1)
channel_per_group = 1

g_off = c_in // channel_per_group
g_c = g_off * kernel * kernel * 2
inpu = torch.FloatTensor([[[[1.0, 2, 3, 4, 1]] * inpu] * c_in] * batchsize).cuda()
offset = torch.FloatTensor([[[[0.000] * out] * out,
                             [[0.000] * out] * out]
                            * kernel * kernel * g_off] * batchsize).cuda()
weight = torch.FloatTensor([[[[1.0, 2, 3]] * kernel] * c_in] * c_out).cuda()
tmp = offset.cpu().numpy()

padding = [pad, pad]
stride = [stri, stri]
dil = [1, 1]
conv_offset = ConvOffset2dFunction(stride, padding, dil, g_off)
output = conv_offset.forward(inpu, offset, weight)
tmp = output.cpu().numpy()

grad_output = torch.FloatTensor([[[[1.0] * out] * out] * c_out] * batchsize).cuda()
columns = torch.zeros(weight.size(1) * weight.size(2) * weight.size(3),
                      inpu.size(0) * output.size(2) * output.size(3)).cuda()
bufs_ = [inpu.new(), inpu.new()]
grad_input = inpu.new(*inpu.size()).zero_()
grad_offset = offset.new(*offset.size()).zero_()
zb.deform_conv_backward_input_cuda(
    inpu, offset, grad_output, grad_input, grad_offset, weight,
    bufs_[0], weight.size(3), weight.size(2),
    stri, stri, pad, pad, 1, 1, g_off
)
grad_input_ = grad_input.cpu().numpy()
grad_offset_ = grad_offset.cpu().numpy()

grad_weight = weight.new(*weight.size()).zero_()
zb.deform_conv_backward_parameters_cuda(
    inpu, offset, grad_output, grad_weight, bufs_[0],
    bufs_[1],
    weight.size(3),
    weight.size(2), stri, stri,
    pad, pad, 1, 1, g_off, 1)
grad_weight_ = grad_weight.cpu().numpy()

print('forward\n', tmp)
print('grad input\n', grad_input_)
print('grad weight\n', grad_weight_)
print('grad offset\n', grad_offset_)
# ---------------------------------my------------------
padding = [pad, pad]
stride = [stri, stri]
conv_offset = my(stride, padding, channel_per_group)
output = conv_offset.forward(inpu, offset, weight)
tmp = output.cpu().numpy()

grad_output = torch.FloatTensor([[[[1.0] * out] * out] * c_out] * batchsize).cuda()
columns = torch.zeros(weight.size(1) * weight.size(2) * weight.size(3),
                      inpu.size(0) * output.size(2) * output.size(3)).cuda()

grad_input = inpu.new(*inpu.size()).zero_()
deform_conv.deform_conv_backward_input_cuda(
    weight, offset, grad_output, columns, grad_input,
    pad, pad, stri, stri, channel_per_group)
grad_input_ = grad_input.cpu().numpy()
grad_offset = offset.new(*offset.size()).zero_()
deform_conv.deform_conv_backward_offset_cuda(
    inpu, weight, offset, grad_output, columns, grad_offset,
    pad, pad, stri, stri, channel_per_group)
grad_offset_ = grad_offset.cpu().numpy()

grad_weight = weight.new(*weight.size()).zero_()
deform_conv.deform_conv_backward_weight_cuda(
    inpu, offset, grad_output, columns, grad_weight,
    pad, pad, stri, stri, channel_per_group)
grad_weight_ = grad_weight.cpu().numpy()

print('forward\n', tmp)
print('grad input\n', grad_input_)
print('grad weight\n', grad_weight_)
print('grad offset\n', grad_offset_)
