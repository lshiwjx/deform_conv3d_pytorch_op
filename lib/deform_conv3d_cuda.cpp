//
// Created by lshi on 17-9-25.
//

#include "deform_conv3d_cuda.h"

inline int input_to_output(int input, int pad, int kernel, int stride) {
    return (input + 2 * pad - kernel) / stride + 1;
}

void shape_check(THCState *state,
                 THCudaTensor *input, THCudaTensor *weight, THCudaTensor *offset, THCudaTensor *output,
                 const int kernel_l, const int kernel_h, const int kernel_w,
                 const int pad_l, const int pad_h, const int pad_w,
                 const int stride_l, const int stride_h, const int stride_w,
                 const int channel_per_deformable_group) {

    int kernel_dim = weight->nDimension;
    long kernel_input_c = weight->size[0];
    long kernel_output_c = weight->size[1];
    long kernel_l = weight->size[2];
    long kernel_h = weight->size[3];
    long kernel_w = weight->size[4];

    int input_dim = input->nDimension;
    long input_b = input->size[0];
    long input_c = input->size[1];
    long input_l = input->size[2];
    long input_h = input->size[3];
    long input_w = input->size[4];

    int output_dim = output->nDimension;
    long output_b = output->size[0];
    long output_c = output->size[1];
    long output_l = output->size[2];
    long output_h = output->size[3];
    long output_w = output->size[4];

    int offset_dim = offset->nDimension;
    long offset_g = offset->size[0];
    long offset_c = offset->size[1];
    long offset_l = offset->size[2];
    long offset_h = offset->size[3];
    long offset_w = offset->size[4];
    //C"CL'H'W'
    THArgCheck(weight->nDimension == 5, 5,
               "5D weight tensor (C\",C, L\',H\',W\') expected, "
                       "but got: %s",
               weight->nDimension);
    THArgCheck(THCudaTensor_isContiguous(state, weight), 5,
               "weight tensor has to be contiguous");
    THArgCheck(kernel_w > 0 && kernel_h > 0 && kernel_l > 0, 9,
               "kernel size should be greater than zero, but got kernel_h: %d kernel_w: %d kernel_l: %d",
               kernel_h, kernel_w, kernel_l);
    THArgCheck((weight->size[2] == kernel_l && weight->size[3] == kernel_h && weight->size[3] == kernel_w), 9,
               "kernel size should be consistent with weight, ",
               "but got kernel_l: %d kernel_h: %d kernel_w: %d weight.size(2): %d, weight.size(3): %d, weight.size(4): %d",
               kernel_l, kernel_h,
               kernel_w, weight->size[2], weight->size[3], weight->size[4]);


    THArgCheck(stride_w > 0 && stride_h > 0 && stride_l > 0, 11,
               "stride should be greater than zero, but got stride_l: %d stride_h: %d stride_w: %d",
               stride_l, stride_h, stride_w);


    THArgCheck(ndim == 4 || ndim == 5, 2,
               "4D or 5D input tensor expected but got: %s", ndim);


    THArgCheck(input_c % deformable_group == 0, 2,
               "input channels must divide deformable group size");

    if (output_w < 1 || output_h < 1 || output_l < 1)
        THError(
                "Given input size: (%ld x %ld x %ld x %ld). "
                        "Calculated output size: (%ld x %ld x %ld x %ld). Output size is too small",
                input_c, input_l, input_h, input_w, nOutputPlane, output_l, output_h,
                output_w);

    THArgCheck(input->size[1] == input_c, 2,
               "invalid number of input planes, expected: %d, but got: %d",
               input_c, input->size[1]);

    THArgCheck((input_l + 2 * padL >= kernel_l && input_h + 2 * padH >= kernel_h && input_w + 2 * padW >= kernel_w), 2,
               "input image is smaller than kernel");

    THArgCheck(
            (offset->size[2] == output_l && offset->size[3] == output_h && offset->size[4] == output_w), 3,
            "invalid spatial size of offset, expected length: %d height: %d width: %d, but got length: %d height: %d width: %d",
            output_l, output_h, output_w,
            offset->size[2], offset->size[3], offset->size[4]);

    THArgCheck((offset->size[1] == deformable_group * 3 * kernel_l * kernel_h * kernel_w), 3,
               "invalid number of channels of offset");

    if (gradOutput != NULL) {
        THArgCheck(gradOutput->size[dimf] == nOutputPlane, 4,
                   "invalid number of gradOutput planes, expected: %d, but got: %d",
                   nOutputPlane, gradOutput->size[dimf]);

        THArgCheck((gradOutput->size[diml] == output_l && gradOutput->size[dimh] == output_h &&
                    gradOutput->size[dimw] == output_w),
                   4,
                   "invalid size of gradOutput, expected length: %d height: %d width: %d , but got length: %d height: %d width: %d",
                   output_l, output_h, output_w,
                   gradOutput->size[diml], gradOutput->size[dimh], gradOutput->size[dimw]);
    }
}
