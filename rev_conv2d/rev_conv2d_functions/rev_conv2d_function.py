import torch
from torch.autograd import Function, Variable

from rev_conv2d import conv2d_op


class Conv2dFunction(Function):
    @staticmethod
    def forward(ctx, inputs, weight1, weight2, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), group=1):
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.group = group
        # ctx.save_for_backward(inputs, weight, bias)
        ctx.c = weight1.size(0)
        # c1 = weight.size(1) // 2
        # c2 = weight.size(1) - c1
        # co1 = weight.size(0) // 2
        # co2 = weight.size(0) - co1
        input1 = inputs[:, :ctx.c, :, :]
        input2 = inputs[:, ctx.c:, :, :]

        output1 = inputs.new(input1.size).zero_()
        output2 = inputs.new(input2.size).zero_()

        ctx.columns = inputs.new(input1.size(1) * weight1.size(2) * weight1.size(3),
                                 output1.size(2) * output1.size(3)).zero_()
        # ctx.columns2 = inputs.new(input2.size(1) * weight2.size(2) * weight2.size(3),
        #                           output_size[0] * output_size[1]).zero_()

        conv2d_op.conv_forward_cuda(
            input2, weight1, ctx.columns, output1,
            ctx.padding[0], ctx.padding[1],
            ctx.stride[0], ctx.stride[1],
            ctx.dilation[0], ctx.dilation[1],
            ctx.group)
        output1 = output1 + input1

        conv2d_op.conv_forward_cuda(
            output1, weight2, ctx.columns, output2,
            ctx.padding[0], ctx.padding[1],
            ctx.stride[0], ctx.stride[1],
            ctx.dilation[0], ctx.dilation[1],
            ctx.group)
        output2 = output2 + input2

        output = torch.cat([output1, output2], 1)

        if bias is not None:
            output += bias.view((1, -1, 1, 1)).expand_as(output)

        return output

    @staticmethod
    def backward(ctx, grad_output, output):
        # inputs, weight, bias = ctx.saved_variables
        grad_output1 = grad_output[:, :ctx.c, :, :]
        grad_output2 = grad_output[:, ctx.c:, :, :]

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
