import deform3dl.deform_conv3dl_op as deform_conv
import torch

from deform3dl.deform_conv3dl_functions import ConvOffset3dFunction

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

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

inpu = torch.DoubleTensor([[[[[1.0, 1, 1], [2.0, 2, 2], [1.0, 1, 1]]] * in_l] * c_in] * batchsize).cuda()
# inpu = torch.randn(1,1,3,7,7).cuda()
offset = torch.DoubleTensor([[[[[0.] * out_w] * out_h] * out_l] * g_off] * batchsize).cuda()
weight = torch.DoubleTensor([[[[[1.0] * kernel_w] * kernel_h] * kernel_l] * c_in] * c_out).cuda()
tmp = offset.cpu().numpy()

padding = [pad, pad, pad]
stride = [stri, stri, stri]
conv_offset = ConvOffset3dFunction(stride, padding, channel_per_group)
output = conv_offset.forward(inpu, offset, weight)
tmp = output.cpu().numpy()

grad_output = torch.DoubleTensor([[[[[1.0] * out_w] * out_h] * out_l] * c_out] * batchsize).cuda()
columns = torch.zeros(weight.size(1) * weight.size(2) * weight.size(3) * weight.size(4),
                      output.size(2) * output.size(3) * output.size(4)).type(torch.DoubleTensor).cuda()

# test for graph and grad
# inputs = Variable(torch.randn(batchsize,c_in,in_l,in_h,in_w)
#                   .type(torch.DoubleTensor).cuda(),requires_grad=True)
# offsets = Variable(torch.randn(batchsize,g_c,out_l,out_h,out_w)
#                    .type(torch.DoubleTensor).cuda(),requires_grad=True)
# weights = Variable(torch.randn(c_out,c_in,kernel_l,kernel_h,kernel_w)
#                    .type(torch.DoubleTensor).cuda(), requires_grad=True)
# o1 = conv_offset(inputs, offsets, weights)
# o2 = Variable(grad_output,requires_grad=False)
# loss = (o1 - o2).sum()
# loss.backward()
# grad_i = inputs.grad.data.cpu().numpy()
# grad_o = offsets.grad.data.cpu().numpy()
# grad_w = weights.grad.data.cpu().numpy()
# ---------------------------------------------

grad_input = inpu.new(*inpu.size()).type(torch.DoubleTensor).zero_()
grad_offset = offset.new(*offset.size()).type(torch.DoubleTensor).zero_()
deform_conv.deform_conv_backward_input_offset_cuda(
    inpu, weight, offset, grad_output, columns, grad_input, grad_offset,
    pad, pad, pad,
    stri, stri, stri, channel_per_group)
grad_input_ = grad_input.cpu().numpy()
grad_offset_ = grad_offset.cpu().numpy()

grad_weight = weight.new(*weight.size()).type(torch.DoubleTensor).zero_()
deform_conv.deform_conv_backward_weight_cuda(
    inpu, offset, grad_output, columns, grad_weight,
    pad, pad, pad,
    stri, stri, stri, channel_per_group)
grad_weight_ = grad_weight.cpu().numpy()

print('forward\n', tmp)
print('grad input\n', grad_input_)
print('grad weight\n', grad_weight_)
print('grad offset\n', grad_offset_)
