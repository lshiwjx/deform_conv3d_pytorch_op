//
// Created by lshi on 17-9-25.
//

//#include "deform_conv3d_cuda.h"
#include "TH/TH.h"
#include "THC/THC.h"
#include "deform_conv3d_cuda_kernel.h"
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
                 THCudaDoubleTensor *input, THCudaDoubleTensor *weight, THCudaDoubleTensor *offset, THCudaDoubleTensor *output,
                 THCudaDoubleTensor *columns,
                 const int pad_l, const int pad_h, const int pad_w,
                 const int stride_l, const int stride_h, const int stride_w,
                 const int channel_per_deformable_group) {
    THCAssertSameGPU(THCudaDoubleTensor_checkGPU(state, 5, input, weight, offset, output, columns));

    int kernel_dim = weight->nDimension;
    check(kernel_dim == 5, "kernel dim");
    long kernel_output_c = weight->size[0];
    long kernel_input_c = weight->size[1];
    long kernel_l = weight->size[2];
    long kernel_h = weight->size[3];
    long kernel_w = weight->size[4];

    int input_dim = input->nDimension;
    check(input_dim == 5, "input dim");
    long input_b = input->size[0];
    long input_c = input->size[1];
    check(input_c == kernel_input_c, "input c != kernel c");
    long input_l = input->size[2];
    long input_h = input->size[3];
    long input_w = input->size[4];

    //out NC"L"H"W"
    int output_dim = output->nDimension;
    check(output_dim == 5, "output dim");
    long output_b = output->size[0];
    check(output_b == input_b, "output b != input b");
    long output_c = output->size[1];
    check(output_c == kernel_output_c, "ou c != ker c");
    long output_l = output->size[2];
    long output_h = output->size[3];
    long output_w = output->size[4];

    int offset_dim = offset->nDimension;
    check(offset_dim == 5, "off dim");
    long offset_b = offset->size[0];
    check(offset_b == input_b, "off batch");
    long offset_c = offset->size[1];
//    cout<<offset_c<<" "<<input_c / channel_per_deformable_group * 3 * kernel_l * kernel_h * kernel_w<<endl;
    check(offset_c == input_c / channel_per_deformable_group, "off channel");
    long offset_l = offset->size[2];
    check(offset_l == output_l, "off l");
    long offset_h = offset->size[3];
    check(offset_h == output_h, "off h");
    long offset_w = offset->size[4];
    check(offset_w == output_w, "off w");

    int columns_dim = columns->nDimension;
    check(columns_dim == 2, "columns dim");
    long columns_row = columns->size[0];
    check(columns_row == input_c * kernel_l * kernel_h * kernel_w, "columns row");
    long columns_col = columns->size[1];
    check(columns_col == output_l * output_h * output_w, "columns col");

    long output_ll = input_to_output(input_l, pad_l, kernel_l, stride_l);
    check(output_ll == output_l, "error output ll");
    long output_hh = input_to_output(input_h, pad_h, kernel_h, stride_h);
    check(output_hh == output_h, "error outhh");
    long output_ww = input_to_output(input_w, pad_w, kernel_w, stride_w);
    check(output_ww == output_w, "error outww");
}


int deform_conv_forward_cuda(
        THCudaDoubleTensor *input, THCudaDoubleTensor *weight, THCudaDoubleTensor *offset,
        THCudaDoubleTensor *columns, THCudaDoubleTensor *output,
        const int pad_l, const int pad_h, const int pad_w,
        const int stride_l, const int stride_h, const int stride_w,
        const int channel_per_deformable_group) {
//    cout<<output->size[0]<<output->size[1]<<output->size[2];
    shape_check(state, input, weight, offset, output, columns,
                pad_l, pad_h, pad_w, stride_l, stride_h, stride_w, channel_per_deformable_group);

    THCudaDoubleTensor *input_n = THCudaDoubleTensor_new(state);
    THCudaDoubleTensor *offset_n = THCudaDoubleTensor_new(state);
    THCudaDoubleTensor *output_n = THCudaDoubleTensor_new(state);

    for(int i=0;i<input->size[0];i++){
        THCudaDoubleTensor_select(state, input_n, input, 0, i);
        THCudaDoubleTensor_select(state, offset_n, offset, 0, i);
        THCudaDoubleTensor_select(state, output_n, output, 0, i);

        deformable_im2col(THCState_getCurrentStream(state),
                          THCudaDoubleTensor_data(state, input_n), THCudaDoubleTensor_data(state, offset_n),
                          input->size[1], input->size[2], input->size[3], input->size[4],
                          output->size[2], output->size[3], output->size[4],
                          weight->size[2], weight->size[3], weight->size[4],
                          pad_l, pad_h, pad_w,
                          stride_l, stride_h, stride_w,
                          channel_per_deformable_group, THCudaDoubleTensor_data(state, columns));
        //GEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
        //C := alpha*op( A )*op( B ) + beta*C,
        int m = weight->size[0];
        int k = columns->size[0];
        int n = columns->size[1];

        THCudaBlas_Dgemm(state, 'n', 'n', n, m, k,
                         1.0f, THCudaDoubleTensor_data(state, columns), n,
                         THCudaDoubleTensor_data(state, weight), k,
                         0.0f, THCudaDoubleTensor_data(state, output_n), n);
        }

    THCudaDoubleTensor_free(state, input_n);
    THCudaDoubleTensor_free(state, offset_n);
    THCudaDoubleTensor_free(state, output_n);

    return 1;
}


int deform_conv_backward_input_offset_cuda(
        THCudaDoubleTensor *input, THCudaDoubleTensor *weight, THCudaDoubleTensor *offset, THCudaDoubleTensor *grad_output,
        THCudaDoubleTensor *columns, THCudaDoubleTensor *grad_input, THCudaDoubleTensor *grad_offset,
        const int pad_l, const int pad_h, const int pad_w,
        const int stride_l, const int stride_h, const int stride_w,
        const int channel_per_deformable_group) {
    shape_check(state, input, weight, grad_offset, grad_output, columns,
                pad_l, pad_h, pad_w, stride_l, stride_h, stride_w, channel_per_deformable_group);
    check(THCudaDoubleTensor_isSameSizeAs(state, offset, grad_offset), "offset vs grad_offset");
    THCAssertSameGPU(THCudaDoubleTensor_checkGPU(state, 2, offset, grad_offset));

    THCudaDoubleTensor *input_n = THCudaDoubleTensor_new(state);
    THCudaDoubleTensor *offset_n = THCudaDoubleTensor_new(state);
    THCudaDoubleTensor *grad_input_n = THCudaDoubleTensor_new(state);
    THCudaDoubleTensor *grad_offset_n = THCudaDoubleTensor_new(state);
    THCudaDoubleTensor *grad_output_n = THCudaDoubleTensor_new(state);

    for(int i=0;i<input->size[0];i++){
        THCudaDoubleTensor_select(state, input_n, input, 0, i);
        THCudaDoubleTensor_select(state, offset_n, offset, 0, i);
        THCudaDoubleTensor_select(state, grad_input_n, grad_input, 0, i);
        THCudaDoubleTensor_select(state, grad_offset_n, grad_offset, 0, i);
        THCudaDoubleTensor_select(state, grad_output_n, grad_output, 0, i);

    //Wt * O = C
    long m = columns->size[0];
    long n = columns->size[1];
    long k = weight->size[0];
    THCudaBlas_Dgemm(state, 'n', 't', n, m, k,
                     1.0f, THCudaDoubleTensor_data(state, grad_output_n), n,
                     THCudaDoubleTensor_data(state, weight), m,
                     0.0f, THCudaDoubleTensor_data(state, columns), n);
    deformable_col2im_offset(THCState_getCurrentStream(state), THCudaDoubleTensor_data(state, columns),
                             THCudaDoubleTensor_data(state, input_n), THCudaDoubleTensor_data(state, offset_n),
                             input->size[1], input->size[2], input->size[3], input->size[4],
                             grad_output->size[2], grad_output->size[3], grad_output->size[4],
                             weight->size[2], weight->size[3], weight->size[4],
                             pad_l, pad_h, pad_w,
                             stride_l, stride_h, stride_w,
                             channel_per_deformable_group, THCudaDoubleTensor_data(state, grad_offset_n));

    deformable_col2im_input(THCState_getCurrentStream(state),
                                THCudaDoubleTensor_data(state, columns),
                                THCudaDoubleTensor_data(state, offset_n),
                                grad_input->size[1], grad_input->size[2], grad_input->size[3], grad_input->size[4],
                                grad_output->size[2], grad_output->size[3], grad_output->size[4],
                                weight->size[2], weight->size[3], weight->size[4],
                                pad_l, pad_h, pad_w,
                                stride_l, stride_h, stride_w, channel_per_deformable_group,
                                THCudaDoubleTensor_data(state, grad_input_n));

    }

    THCudaDoubleTensor_free(state, input_n);
    THCudaDoubleTensor_free(state, offset_n);
    THCudaDoubleTensor_free(state, grad_input_n);
    THCudaDoubleTensor_free(state, grad_offset_n);
    THCudaDoubleTensor_free(state, grad_output_n);

    return 1;
}


int deform_conv_backward_weight_cuda(
        THCudaDoubleTensor *input, THCudaDoubleTensor *offset, THCudaDoubleTensor *grad_output,
        THCudaDoubleTensor *columns, THCudaDoubleTensor *grad_weight,
        const int pad_l, const int pad_h, const int pad_w,
        const int stride_l, const int stride_h, const int stride_w,
        const int channel_per_deformable_group) {
    shape_check(state, input, grad_weight, offset, grad_output, columns,
                pad_l, pad_h, pad_w, stride_l, stride_h, stride_w, channel_per_deformable_group);
    THCudaDoubleTensor *input_n = THCudaDoubleTensor_new(state);
    THCudaDoubleTensor *offset_n = THCudaDoubleTensor_new(state);
    THCudaDoubleTensor *grad_output_n = THCudaDoubleTensor_new(state);

    for(int i=0;i<input->size[0];i++){
        THCudaDoubleTensor_select(state, input_n, input, 0, i);
        THCudaDoubleTensor_select(state, offset_n, offset, 0, i);
        THCudaDoubleTensor_select(state, grad_output_n, grad_output, 0, i);

        deformable_im2col(THCState_getCurrentStream(state),
                          THCudaDoubleTensor_data(state, input_n), THCudaDoubleTensor_data(state, offset_n),
                          input->size[1], input->size[2], input->size[3], input->size[4],
                          grad_output->size[2], grad_output->size[3], grad_output->size[4],
                          grad_weight->size[2], grad_weight->size[3], grad_weight->size[4],
                          pad_l, pad_h, pad_w,
                          stride_l, stride_h, stride_w,
                          channel_per_deformable_group,
                          THCudaDoubleTensor_data(state, columns));
        //GEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
        //C := alpha*op( A )*op( B ) + beta*C,
        int m = grad_weight->size[0];
        int k = columns->size[1];
        int n = columns->size[0];
        THCudaBlas_Dgemm(state, 't', 'n', n, m, k,
                         1.0f, THCudaDoubleTensor_data(state, columns), k,
                         THCudaDoubleTensor_data(state, grad_output_n), k,
                         1.0f, THCudaDoubleTensor_data(state, grad_weight), n);
    }

    THCudaDoubleTensor_free(state, input_n);
    THCudaDoubleTensor_free(state, offset_n);
    THCudaDoubleTensor_free(state, grad_output_n);

    return 1;
}
