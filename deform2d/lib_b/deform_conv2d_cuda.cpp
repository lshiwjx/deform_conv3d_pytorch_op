//
// Created by lshi on 17-9-25.
//

//#include "deform_conv2d_cuda.h"
#include <THC/THCStorage.h>
#include "TH/TH.h"
#include "THC/THC.h"
#include "deform_conv2d_cuda_kernel.h"
#include "iostream"

using namespace std;
extern THCState *state;

inline int input_to_output(int input, int pad, int kernel, int stride) {
    return (input + 2 * pad - kernel) / stride + 1;
}

void check(bool e, const char *msg) {
    if (!e)
        THError(msg);
}

void shape_check(THCState *state,
                 THCudaTensor *input, THCudaTensor *weight, THCudaTensor *offset, THCudaTensor *output,
                 THCudaTensor *columns,
                 const int pad_h, const int pad_w,
                 const int stride_h, const int stride_w,
                 const int channel_per_deformable_group) {
    THCAssertSameGPU(THCudaTensor_checkGPU(state, 5, input, weight, offset, output, columns));

    int kernel_dim = weight->nDimension;
    check(kernel_dim == 4, "kernel dim");
    long kernel_output_c = weight->size[0];
    long kernel_input_c = weight->size[1];
    long kernel_h = weight->size[2];
    long kernel_w = weight->size[3];

    int input_dim = input->nDimension;
    check(input_dim == 4, "input dim");
    long input_b = input->size[0];
    long input_c = input->size[1];
    check(input_c == kernel_input_c, "input c != kernel c");
    long input_h = input->size[2];
    long input_w = input->size[3];

    //out NC"L"H"W"
    int output_dim = output->nDimension;
    check(output_dim == 4, "output dim");
    long output_b = output->size[0];
    check(output_b == input_b, "output b != input b");
    long output_c = output->size[1];
    check(output_c == kernel_output_c, "ou c != ker c");
    long output_h = output->size[2];
    long output_w = output->size[3];

    int offset_dim = offset->nDimension;
    check(offset_dim == 4, "off dim");
    long offset_b = offset->size[0];
    check(offset_b == input_b, "off batch");
    long offset_c = offset->size[1];
    check(offset_c == input_c / channel_per_deformable_group * 2 * kernel_h * kernel_w, "off channel");
    long offset_h = offset->size[2];
    check(offset_h == output_h, "off h");
    long offset_w = offset->size[3];
    check(offset_w == output_w, "off w");

    int columns_dim = columns->nDimension;
    check(columns_dim == 2, "columns dim");
    long columns_row = columns->size[0];
    check(columns_row == input_c * kernel_h * kernel_w, "columns row");
    long columns_col = columns->size[1];
    check(columns_col == input_b * output_h * output_w, "columns col");

    long output_hh = input_to_output(input_h, pad_h, kernel_h, stride_h);
    check(output_hh == output_h, "error outhh");
    long output_ww = input_to_output(input_w, pad_w, kernel_w, stride_w);
    check(output_ww == output_w, "error outww");
}


int deform_conv_forward_cuda(
        THCudaTensor *input, THCudaTensor *weight, THCudaTensor *offset,
        THCudaTensor *columns, THCudaTensor *output,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int channel_per_deformable_group) {
//    cout<<"safe"<<endl;
    shape_check(state, input, weight, offset, output, columns,
                pad_h, pad_w, stride_h, stride_w, channel_per_deformable_group);
    deformable_im2col(THCState_getCurrentStream(state),
                      THCudaTensor_data(state, input), THCudaTensor_data(state, offset),
                      input->size[0], input->size[1], input->size[2], input->size[3],
                      output->size[2], output->size[3],
                      weight->size[2], weight->size[3],
                      pad_h, pad_w,
                      stride_h, stride_w,
                      channel_per_deformable_group, THCudaTensor_data(state, columns));
    //GEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
    //C := alpha*op( A )*op( B ) + beta*C,
    int m = weight->size[0];
    int k = columns->size[0];
    int n = columns->size[1];
//    cout<<"safe"<<output->size[0]<<output->size[1]<<output->size[2]<<output->size[3]<<endl;
    THCudaTensor_transpose(state, output, output, 0, 1);
//    cout<<"safe"<<output->size[0]<<output->size[1]<<output->size[2]<<output->size[3]<<endl;
    THCudaBlas_Sgemm(state, 'n', 'n', n, m, k,
                     1.0f, THCudaTensor_data(state, columns), n,
                     THCudaTensor_data(state, weight), k,
                     0.0f, THCudaTensor_data(state, output), n);
//    cout<<"safe"<<output->size[0]<<output->size[1]<<output->size[2]<<output->size[3]<<endl;
    THCudaTensor_transpose(state, output, output, 0, 1);
//    cout<<"safe"<<output->size[0]<<output->size[1]<<output->size[2]<<output->size[3]<<endl;
    return 1;
}


int deform_conv_backward_input_cuda(
        THCudaTensor *weight, THCudaTensor *offset, THCudaTensor *grad_output,
        THCudaTensor *columns, THCudaTensor *grad_input,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int channel_per_deformable_group) {
    shape_check(state, grad_input, weight, offset, grad_output, columns,
                pad_h, pad_w, stride_h, stride_w, channel_per_deformable_group);

    //Wt * O = C
    long m = columns->size[0];
    long n = columns->size[1];
    long k = weight->size[0];
    THCudaTensor_transpose(state, grad_output, grad_output, 0, 1);

    THCudaBlas_Sgemm(state, 'n', 't', n, m, k,
                     1.0f, THCudaTensor_data(state, grad_output), n,
                     THCudaTensor_data(state, weight), m,
                     0.0f, THCudaTensor_data(state, columns), n);
    THCudaTensor_transpose(state, grad_output, grad_output, 0, 1);
    deformable_col2im_input(THCState_getCurrentStream(state),
                            THCudaTensor_data(state, columns),
                            THCudaTensor_data(state, offset),
                            grad_input->size[0], grad_input->size[1],
                            grad_input->size[2], grad_input->size[3],
                            grad_output->size[2], grad_output->size[3],
                            weight->size[2], weight->size[3],
                            pad_h, pad_w,
                            stride_h, stride_w, channel_per_deformable_group,
                            THCudaTensor_data(state, grad_input));

    return 1;
}

int deform_conv_backward_offset_cuda(
        THCudaTensor *input, THCudaTensor *weight, THCudaTensor *offset, THCudaTensor *grad_output,
        THCudaTensor *columns, THCudaTensor *grad_offset,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int channel_per_deformable_group) {
    shape_check(state, input, weight, grad_offset, grad_output, columns,
                pad_h, pad_w, stride_h, stride_w, channel_per_deformable_group);
    check(THCudaTensor_isSameSizeAs(state, offset, grad_offset), "offset vs grad_offset");
    THCAssertSameGPU(THCudaTensor_checkGPU(state, 2, offset, grad_offset));
    //Wt * O = C
    long m = columns->size[0];
    long n = columns->size[1];
    long k = weight->size[0];
    THCudaTensor_transpose(state, grad_output, grad_output, 0, 1);
    THCudaBlas_Sgemm(state, 'n', 't', n, m, k,
                     1.0f, THCudaTensor_data(state, grad_output), n,
                     THCudaTensor_data(state, weight), m,
                     0.0f, THCudaTensor_data(state, columns), n);
    THCudaTensor_transpose(state, grad_output, grad_output, 0, 1);
    deformable_col2im_offset(THCState_getCurrentStream(state), THCudaTensor_data(state, columns),
                             THCudaTensor_data(state, input), THCudaTensor_data(state, offset),
                             input->size[0], input->size[1], input->size[2], input->size[3],
                             grad_output->size[2], grad_output->size[3],
                             weight->size[2], weight->size[3],
                             pad_h, pad_w,
                             stride_h, stride_w,
                             channel_per_deformable_group, THCudaTensor_data(state, grad_offset));

    return 1;
}


int deform_conv_backward_weight_cuda(
        THCudaTensor *input, THCudaTensor *offset, THCudaTensor *grad_output,
        THCudaTensor *columns, THCudaTensor *grad_weight,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int channel_per_deformable_group) {
    shape_check(state, input, grad_weight, offset, grad_output, columns,
                pad_h, pad_w, stride_h, stride_w, channel_per_deformable_group);

    deformable_im2col(THCState_getCurrentStream(state),
                      THCudaTensor_data(state, input), THCudaTensor_data(state, offset),
                      input->size[0], input->size[1], input->size[2], input->size[3],
                      grad_output->size[2], grad_output->size[3],
                      grad_weight->size[2], grad_weight->size[3],
                      pad_h, pad_w,
                      stride_h, stride_w,
                      channel_per_deformable_group,
                      THCudaTensor_data(state, columns));
    //GEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
    //C := alpha*op( A )*op( B ) + beta*C,
    int m = grad_weight->size[0];
    int k = columns->size[1];
    int n = columns->size[0];
    THCudaTensor_transpose(state, grad_output, grad_output, 0, 1);
    THCudaBlas_Sgemm(state, 't', 'n', n, m, k,
                     1.0f, THCudaTensor_data(state, columns), k,
                     THCudaTensor_data(state, grad_output), k,
                     0.0f, THCudaTensor_data(state, grad_weight), n);
    THCudaTensor_transpose(state, grad_output, grad_output, 0, 1);
    return 1;
}
