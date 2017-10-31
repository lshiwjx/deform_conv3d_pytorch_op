import torch
from torch.autograd import Function

from deform2d import deform_conv2d_op


class ConvOffset2dFunction(Function):
    def __init__(self, stride, padding, channel_per_group):
        super(ConvOffset2dFunction, self).__init__()
        self.stride = stride
        self.padding = padding
        self.channel_per_group = channel_per_group
        self.savedtensors = ()

    def forward(self, input, offset, weight):
        self.save_for_backward(input, offset, weight)

        output = torch.zeros(input.size(0), weight.size(0),
                             self._to_output(input.size(2), weight.size(2), self.padding[0], self.stride[0]),
                             self._to_output(input.size(3), weight.size(3), self.padding[1], self.stride[1])) \
            .type(torch.FloatTensor).cuda()
        columns = torch.zeros(weight.size(1) * weight.size(2) * weight.size(3),
                              output.size(2) * output.size(3)).type(torch.FloatTensor).cuda()

        if not input.is_cuda:
            raise NotImplementedError
        else:
            if not isinstance(input, torch.cuda.FloatTensor):
                raise NotImplementedError
            deform_conv2d_op.deform_conv_forward_cuda(
                input, weight, offset, columns, output,
                self.padding[0], self.padding[1],
                self.stride[0], self.stride[1],
                self.channel_per_group)

        return output

    def backward(self, grad_output):
        input, offset, weight = self.saved_tensors

        grad_input = grad_offset = grad_weight = None

        if not grad_output.is_cuda:
            raise NotImplementedError
        else:
            if not isinstance(grad_output, torch.cuda.FloatTensor):
                raise NotImplementedError
            columns = torch.zeros(weight.size(1) * weight.size(2) * weight.size(3),
                                  grad_output.size(2) * grad_output.size(3)) \
                .type(torch.FloatTensor).cuda()
            if self.needs_input_grad[0]:
                grad_input = torch.zeros(*input.size()).type(torch.FloatTensor).cuda()

                deform_conv2d_op.deform_conv_backward_input_cuda(
                    weight, offset, grad_output, columns, grad_input,
                    self.padding[0], self.padding[1],
                    self.stride[0], self.stride[1],
                    self.channel_per_group)

            if self.needs_input_grad[1]:
                grad_offset = torch.zeros(*offset.size()).type(torch.FloatTensor).cuda()

                deform_conv2d_op.deform_conv_backward_offset_cuda(
                    input, weight, offset, grad_output, columns, grad_offset,
                    self.padding[0], self.padding[1],
                    self.stride[0], self.stride[1],
                    self.channel_per_group)

            if self.needs_input_grad[2]:
                grad_weight = torch.zeros(*weight.size()).type(torch.FloatTensor).cuda()

                deform_conv2d_op.deform_conv_backward_weight_cuda(
                    input, offset, grad_output, columns, grad_weight,
                    self.padding[0], self.padding[1],
                    self.stride[0], self.stride[1],
                    self.channel_per_group)

        return grad_input, grad_offset, grad_weight

    def _to_output(self, inpu, kernel, pad, stride):
        return int((inpu + 2 * pad - kernel) / stride + 1)
