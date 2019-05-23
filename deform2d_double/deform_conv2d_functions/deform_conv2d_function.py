import torch
from torch.autograd import Function, Variable

import deform_conv2d_op


class ConvOffset2dFunction(Function):
    # @staticmethod
    # def init(ctx, stride, padding, channel_per_group):
    #     ctx.stride = stride
    #     ctx.padding = padding
    #     ctx.channel_per_group = channel_per_group
    #     ctx.savedtensors = ()

    @staticmethod
    def forward(ctx, inputs, offset, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1),
                channel_per_group=1, group=1):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.channel_per_group = channel_per_group
        ctx.group = group
        ctx.save_for_backward(inputs, offset, weight, bias)

        output_size = [int((inputs.size()[i + 2] + 2 * ctx.padding[i] - weight.size()[i + 2]) / ctx.stride[i] + 1)
                       for i in range(2)]

        output = inputs.new(inputs.size(0), weight.size(0), output_size[0], output_size[1]).zero_()

        ctx.columns = inputs.new(inputs.size(1) * weight.size(2) * weight.size(3),
                                 output_size[0] * output_size[1]).zero_()
        # ctx.ones = inputs.new(output_size[0], output_size[1]).fill_(1)

        deform_conv2d_op.deform_conv_forward_cuda(
            inputs, weight, offset, ctx.columns, output,
            ctx.padding[0], ctx.padding[1],
            ctx.stride[0], ctx.stride[1],
            ctx.dilation[0], ctx.dilation[1],
            ctx.channel_per_group, ctx.group)

        if bias is not None:
            output += bias.view((1, -1, 1, 1)).expand_as(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs, offset, weight, bias = ctx.saved_variables

        grad_input = grad_offset = grad_weight = grad_bias = None
        # 前传得到的是tensor 反传是variable
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grad_input = inputs.data.new(inputs.size()).zero_()
            grad_offset = offset.data.new(offset.size()).zero_()

            deform_conv2d_op.deform_conv_backward_input_offset_cuda(
                inputs.data, weight.data, offset.data, grad_output.data, ctx.columns, grad_input, grad_offset,
                ctx.padding[0], ctx.padding[1],
                ctx.stride[0], ctx.stride[1],
                ctx.dilation[0], ctx.dilation[1],
                ctx.channel_per_group, ctx.group)

        if ctx.needs_input_grad[2]:
            grad_weight = weight.data.new(weight.size()).zero_()

            deform_conv2d_op.deform_conv_backward_weight_cuda(
                inputs.data, offset.data, grad_output.data, ctx.columns, grad_weight,
                ctx.padding[0], ctx.padding[1],
                ctx.stride[0], ctx.stride[1],
                ctx.dilation[0], ctx.dilation[1],
                ctx.channel_per_group, ctx.group)

        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).sum(1).sum(1)

        return Variable(grad_input), Variable(grad_offset), Variable(
            grad_weight), grad_bias, None, None, None, None, None
