import deform2d.deform_conv2d_op as deform_conv
import deform2d._ext.deform_conv as zb
import torch
import os
from deform2d.deform_conv2d_functions import ConvOffset2dFunction as my
from deform2d.functions.deform_conv import ConvOffset2dFunction
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
batchsize = 200
c_in = 16
c_out = 16
inpu = 100
kernel = 3
stri = 1
pad = 0
out = int((inpu + 2 * pad - kernel) / stri + 1)
channel_per_group = 1

g_off = c_in // channel_per_group
g_c = g_off * kernel * kernel * 2
# inpu = torch.FloatTensor([[[[1.0]* inpu] * inpu] * c_in] * batchsize).cuda()
# offset = torch.FloatTensor([[[[0.000] * out] * out,
#                              [[0.000] * out] * out]
#                             * kernel * kernel * g_off] * batchsize).cuda()
# weight = torch.FloatTensor([[[[1.0]* kernel] * kernel] * c_in] * c_out).cuda()
inpu = torch.rand(batchsize, c_in, inpu, inpu).type(torch.FloatTensor).cuda()
offset = torch.rand(batchsize, g_c, out, out).type(torch.FloatTensor).cuda()
weight = torch.rand(c_out, c_in, kernel, kernel).type(torch.FloatTensor).cuda()

# padding = [pad, pad]
# stride = [stri, stri]
# dil = [1, 1]
# start = time.time()
# conv_offset = ConvOffset2dFunction(stride, padding, dil, g_off)
# output = conv_offset.forward(inpu, offset, weight)
# s11 = time.time()
# grad_output = torch.FloatTensor([[[[1.0] * out] * out] * c_out] * batchsize).cuda()
# bufs_ = [inpu.new(), inpu.new()]
# grad_input = inpu.new(*inpu.size()).zero_()
# grad_offset = offset.new(*offset.size()).zero_()
# zb.deform_conv_backward_input_cuda(
#     inpu, offset, grad_output, grad_input, grad_offset, weight,
#     bufs_[0], weight.size(3), weight.size(2),
#     stri, stri, pad, pad, 1, 1, g_off
# )
# s12 = time.time()
# grad_weight = weight.new(*weight.size()).zero_()
# zb.deform_conv_backward_parameters_cuda(
#     inpu, offset, grad_output, grad_weight, bufs_[0],
#     bufs_[1],
#     weight.size(3),
#     weight.size(2), stri, stri,
#     pad, pad, 1, 1, g_off, 1)
# s13 = time.time()
# print('forward\n', tmp)
# print('grad input\n', grad_input_)
# print('grad weight\n', grad_weight_)
# print('grad offset\n', grad_offset_)
# ---------------------------------my------------------
start2 = time.time()
padding = [pad, pad]
stride = [stri, stri]
conv_offset = my(stride, padding, channel_per_group)
output = conv_offset.forward(inpu, offset, weight)
# tmp = output.cpu().numpy()
s21 = time.time()
grad_output = torch.FloatTensor([[[[1.0] * out] * out] * c_out] * batchsize).cuda()
columns = torch.zeros(weight.size(1) * weight.size(2) * weight.size(3),
                      output.size(2) * output.size(3)).cuda()

grad_input = inpu.new(*inpu.size()).zero_()
grad_offset = offset.new(*offset.size()).zero_()
deform_conv.deform_conv_backward_input_offset_cuda(
    inpu, weight, offset, grad_output, columns, grad_input, grad_offset,
    pad, pad, stri, stri, channel_per_group)
# grad_input_ = grad_input.cpu().numpy()
s22 = time.time()
# deform_conv.deform_conv_backward_offset_cuda(
#     inpu, weight, offset, grad_output, columns, grad_offset,
#     pad, pad, stri, stri, channel_per_group)
# grad_offset_ = grad_offset.cpu().numpy()
s23 = time.time()
grad_weight = weight.new(*weight.size()).zero_()
deform_conv.deform_conv_backward_weight_cuda(
    inpu, offset, grad_output, columns, grad_weight,
    pad, pad, stri, stri, channel_per_group)
# grad_weight_ = grad_weight.cpu().numpy()
s24 = time.time()
# print('forward\n', tmp)
# print('grad input\n', grad_input_)
# print('grad weight\n', grad_weight_)
# print('grad offset\n', grad_offset_)
print('1zb:\n forward: ', s11 - start, '\n back in and off: ', s12 - s11, '\n back weight ', s13 - s12)
print('my:\n for: ', s21 - start2, '\n back in: ', s22 - s21, '\n back off ', s23 - s22, '\n back weight ', s24 - s23)
