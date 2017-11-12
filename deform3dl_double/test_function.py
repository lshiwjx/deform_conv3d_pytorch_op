import deform3dl_double.deform_conv3dl_op as deform_conv
import torch
import os
from deform3dl_double.deform_conv3dl_functions import ConvOffset3dFunction

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

# inpu = torch.cuda.DoubleTensor([[[[[1.0, 1, 1], [1.0, 2, 3], [1.0, 3, 1]]] * inpu] * c_in] * batchsize)
inpu = torch.cuda.DoubleTensor(
    [[[[[1.0] * inpu] * inpu, [[2.0] * inpu] * inpu, [[1.0] * inpu] * inpu]] * c_in] * batchsize)
offset = torch.cuda.DoubleTensor([[[[[2.] * out] * out] * out] * g_off] * batchsize)
weight = torch.cuda.DoubleTensor([[[[[1.0] * kernel] * kernel] * kernel] * c_in] * c_out)
tmp = offset.cpu().numpy()

padding = [pad, pad, pad]
stride = [stri, stri, stri]
conv_offset = ConvOffset3dFunction(stride, padding, channel_per_group)
output = conv_offset.forward(inpu, offset, weight)
tmp = output.cpu().numpy()

grad_output = torch.cuda.DoubleTensor([[[[[1.0] * out] * out] * out] * c_out] * batchsize)
columns = torch.zeros(weight.size(1) * weight.size(2) * weight.size(3) * weight.size(4),
                      output.size(2) * output.size(3) * output.size(4)).double().cuda()

# test for graph and grad
# inputs = Variable(torch.randn(batchsize,c_in,in_l,in_h,in_w)
#                   .double().cuda(),requires_grad=True)
# offsets = Variable(torch.randn(batchsize,g_c,out_l,out_h,out_w)
#                    .double().cuda(),requires_grad=True)
# weights = Variable(torch.randn(c_out,c_in,kernel_l,kernel_h,kernel_w)
#                    .double().cuda(), requires_grad=True)
# o1 = conv_offset(inputs, offsets, weights)
# o2 = Variable(grad_output,requires_grad=False)
# loss = (o1 - o2).sum()
# loss.backward()
# grad_i = inputs.grad.data.cpu().numpy()
# grad_o = offsets.grad.data.cpu().numpy()
# grad_w = weights.grad.data.cpu().numpy()
# ---------------------------------------------

grad_input = inpu.new(*inpu.size()).double().zero_()
grad_offset = offset.new(*offset.size()).double().zero_()
deform_conv.deform_conv_backward_input_offset_cuda(
    inpu, weight, offset, grad_output, columns, grad_input, grad_offset,
    pad, pad, pad,
    stri, stri, stri, channel_per_group)
grad_input_ = grad_input.cpu().numpy()
grad_offset_ = grad_offset.cpu().numpy()

grad_weight = weight.new(*weight.size()).double().zero_()
deform_conv.deform_conv_backward_weight_cuda(
    inpu, offset, grad_output, columns, grad_weight,
    pad, pad, pad,
    stri, stri, stri, channel_per_group)
grad_weight_ = grad_weight.cpu().numpy()

print('forward\n', tmp)
print('grad input\n', grad_input_)
# print('grad weight\n', grad_weight_)
print('grad offset\n', grad_offset_)
