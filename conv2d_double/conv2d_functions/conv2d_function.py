import torch
from torch.autograd import Function, Variable

from conv2d_double import conv2d_op


class Conv2dFunction(Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), group=1):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.group = group
        ctx.save_for_backward(inputs, weight, bias)

        output_size = [int(
            (inputs.size()[i + 2] + 2 * ctx.padding[i] - dilation[i] * (weight.size()[i + 2] - 1) - 1) / ctx.stride[
                i] + 1)
            for i in range(2)]

        output = inputs.new(inputs.size(0), weight.size(0), output_size[0], output_size[1]).zero_()

        ctx.columns = inputs.new(inputs.size(1) * weight.size(2) * weight.size(3),
                                 output_size[0] * output_size[1]).zero_()

        conv2d_op.conv_forward_cuda(
            inputs, weight, ctx.columns, output,
            ctx.padding[0], ctx.padding[1],
            ctx.stride[0], ctx.stride[1],
            ctx.dilation[0], ctx.dilation[1],
            ctx.group)

        if bias is not None:
            output += bias.view((1, -1, 1, 1)).expand_as(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight, bias = ctx.saved_variables

        grad_input = grad_weight = grad_bias = None
        # 前传得到的是tensor 反传是variable
        if ctx.needs_input_grad[0]:
            grad_input = inputs.data.new(inputs.size()).zero_()

            conv2d_op.conv_backward_input_cuda(
                weight.data, grad_output.data, ctx.columns, grad_input,
                ctx.padding[0], ctx.padding[1],
                ctx.stride[0], ctx.stride[1],
                ctx.dilation[0], ctx.dilation[1],
                ctx.group)

        if ctx.needs_input_grad[1]:
            grad_weight = weight.data.new(weight.size()).zero_()

            conv2d_op.conv_backward_weight_cuda(
                inputs.data, grad_output.data, ctx.columns, grad_weight,
                ctx.padding[0], ctx.padding[1],
                ctx.stride[0], ctx.stride[1],
                ctx.dilation[0], ctx.dilation[1],
                ctx.group)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).sum(1).sum(1)

        return Variable(grad_input), Variable(grad_weight), grad_bias, None, None, None, None, None
