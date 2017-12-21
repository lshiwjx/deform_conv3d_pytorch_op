import deform3d.deform_conv3d_op as deform_conv
import torch
from torch.autograd import Variable
from deform3d.deform_conv3d_functions import ConvOffset3dFunction

batchsize = 1
c_in = 1
c_out = 1
channel_per_group = 1
kernel_l = kernel_h = kernel_w = 3
out_l = out_h = out_w = 3
in_l = in_h = in_w = 3
pad = 1
stri = 1
dilaton = 2
g_off = c_in // channel_per_group
g_c = g_off * kernel_l * kernel_h * kernel_w * 3
inpu = torch.FloatTensor([[[[[1.0, 1, 1], [2.0, 2, 2], [1.0, 1, 1]]] * in_l] * c_in] * batchsize).cuda()
# inpu = torch.randn(1,1,3,7,7).cuda()
offset = torch.FloatTensor([[[[[0.000] * out_w] * out_h] * out_l,
                             [[[0.000] * out_w] * out_h] * out_l,
                             [[[0.0] * out_w] * out_h] * out_l]
                            * kernel_l * kernel_h * kernel_w * g_off] * batchsize).cuda()
weight = torch.FloatTensor([[[[[1.0] * kernel_w] * kernel_h] * kernel_l] * c_in] * c_out).cuda()
tmp = offset.cpu().numpy()

padding = [pad, pad, pad]
stride = [stri, stri, stri]
dilaton = [dilaton, dilaton, dilaton]
# conv_offset = ConvOffset3dFunction(stride, padding, channel_per_group)
output = ConvOffset3dFunction.apply(inpu, offset, weight, None, stride, padding, dilaton, channel_per_group)
tmp = output.data.cpu().numpy()
print(tmp)
grad_output = torch.FloatTensor([[[[[1.0] * out_w] * out_h] * out_l] * c_out] * batchsize).cuda()
columns = torch.zeros(weight.size(1) * weight.size(2) * weight.size(3) * weight.size(4),
                      output.size(2) * output.size(3) * output.size(4)).cuda()

# test for graph and grad
# inputs = Variable(torch.randn(batchsize,c_in,in_l,in_h,in_w)
#                   .type(torch.FloatTensor).cuda(),requires_grad=True)
# offsets = Variable(torch.randn(batchsize,g_c,out_l,out_h,out_w)
#                    .type(torch.FloatTensor).cuda(),requires_grad=True)
# weights = Variable(torch.randn(c_out,c_in,kernel_l,kernel_h,kernel_w)
#                    .type(torch.FloatTensor).cuda(), requires_grad=True)
# o1 = conv_offset(inputs, offsets, weights)
# o2 = Variable(grad_output,requires_grad=False)
# loss = (o1 - o2).sum()
# loss.backward()
# grad_i = inputs.grad.data.cpu().numpy()
# grad_o = offsets.grad.data.cpu().numpy()
# grad_w = weights.grad.data.cpu().numpy()
# ---------------------------------------------

grad_input = inpu.new(*inpu.size()).zero_()
grad_offset = offset.new(*offset.size()).zero_()
deform_conv.deform_conv_backward_input_offset_cuda(
    inpu, weight, offset, grad_output, columns, grad_input, grad_offset,
    pad, pad, pad,
    stri, stri, stri, channel_per_group)
grad_input_ = grad_input.cpu().numpy()
grad_offset_ = grad_offset.cpu().numpy()

grad_weight = weight.new(*weight.size()).zero_()
deform_conv.deform_conv_backward_weight_cuda(
    inpu, offset, grad_output, columns, grad_weight,
    pad, pad, pad,
    stri, stri, stri, channel_per_group)
grad_weight_ = grad_weight.cpu().numpy()

print('forward\n', tmp)
print('grad input\n', grad_input_)
print('grad weight\n', grad_weight_)
print('grad offset\n', grad_offset_)
