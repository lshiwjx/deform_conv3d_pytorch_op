import torch
from torch.autograd import Function
from torch.nn.modules.utils import _triple

from deform3dl_double import deform_conv3dl_op


class ConvOffset3dFunction(Function):
    def __init__(self, stride, padding, channel_per_group):
        super(ConvOffset3dFunction, self).__init__()
        self.stride = stride
        self.padding = padding
        self.channel_per_group = channel_per_group
        self.savedtensors = ()

    def forward(self, inputs, offset, weight):
        self.save_for_backward(inputs, offset, weight)
        output_size = [int((inputs.size()[i + 2] + 2 * self.padding[i] - weight.size()[i + 2]) / self.stride[i] + 1)
                       for i in range(3)]
        output = inputs.new(inputs.size(0), weight.size(0), output_size[0], output_size[1], output_size[2], ).zero_()

        self.columns = inputs.new(weight.size(1) * weight.size(2) * weight.size(3) * weight.size(4),
                                  output_size[0] * output_size[1] * output_size[2]).zero_()

        deform_conv3dl_op.deform_conv_forward_cuda(
            inputs, weight, offset, self.columns, output,
            self.padding[0], self.padding[1], self.padding[2],
            self.stride[0], self.stride[1], self.stride[2],
            self.channel_per_group)

        return output

    def backward(self, grad_output):
        inputs, offset, weight = self.saved_tensors

        grad_input = grad_offset = grad_weight = None

        if self.needs_input_grad[0] or self.needs_input_grad[1]:
            grad_input = inputs.new(inputs.size()).zero_()
            grad_offset = offset.new(offset.size()).zero_()

            deform_conv3dl_op.deform_conv_backward_input_offset_cuda(
                inputs, weight, offset, grad_output, self.columns, grad_input, grad_offset,
                self.padding[0], self.padding[1], self.padding[2],
                self.stride[0], self.stride[1], self.stride[2],
                self.channel_per_group)

        if self.needs_input_grad[2]:
            grad_weight = weight.new(weight.size()).zero_()

            deform_conv3dl_op.deform_conv_backward_weight_cuda(
                inputs, offset, grad_output, self.columns, grad_weight,
                self.padding[0], self.padding[1], self.padding[2],
                self.stride[0], self.stride[1], self.stride[2],
                self.channel_per_group)

        return grad_input, grad_offset, grad_weight
