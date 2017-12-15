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
                 THCudaDoubleTensor *input, THCudaDoubleTensor *weight, THCudaDoubleTensor *offset, THCudaDoubleTensor *output,
                 THCudaDoubleTensor *columns,
                 const int pad_h, const int pad_w,
                 const int stride_h, const int stride_w,
                 const int channel_per_deformable_group, const int group) {
    THCAssertSameGPU(THCudaDoubleTensor_checkGPU(state, 5, input, weight, offset, output, columns));

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
    check(input_c == kernel_input_c*group, "input c != kernel c");
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
    check(columns_col == output_h * output_w, "columns col");

    long output_hh = input_to_output(input_h, pad_h, kernel_h, stride_h);
    check(output_hh == output_h, "error outhh");
    long output_ww = input_to_output(input_w, pad_w, kernel_w, stride_w);
    check(output_ww == output_w, "error outww");
}


int deform_conv_forward_cuda(
        THCudaDoubleTensor *input, THCudaDoubleTensor *weight, THCudaDoubleTensor *offset,
        THCudaDoubleTensor *columns, THCudaDoubleTensor *output,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        const int channel_per_deformable_group, const int group) {
//    cout<<"safe"<<endl;
    shape_check(state, input, weight, offset, output, columns,
                pad_h, pad_w, stride_h, stride_w, channel_per_deformable_group, group);
    
    THCudaDoubleTensor *input_n = THCudaDoubleTensor_new(state);
    THCudaDoubleTensor *offset_n = THCudaDoubleTensor_new(state);
    THCudaDoubleTensor *output_n = THCudaDoubleTensor_new(state);
    
    for(int i=0;i<input->size[0];i++){
        THCudaDoubleTensor_select(state, input_n, input, 0, i);
        THCudaDoubleTensor_select(state, offset_n, offset, 0, i);
        THCudaDoubleTensor_select(state, output_n, output, 0, i);
        
        deformable_im2col(THCState_getCurrentStream(state),
                      THCudaDoubleTensor_data(state, input_n), THCudaDoubleTensor_data(state, offset_n),
                      input->size[1], input->size[2], input->size[3],
                      output->size[2], output->size[3],
                      weight->size[2], weight->size[3],
                      pad_h, pad_w,
                      stride_h, stride_w,
                      dilation_h, dilation_w,
                      channel_per_deformable_group, THCudaDoubleTensor_data(state, columns));

        THCudaDoubleTensor *columns_g = THCudaDoubleTensor_new(state);
        THCudaDoubleTensor *weight_g = THCudaDoubleTensor_new(state);
        THCudaDoubleTensor *output_n_g = THCudaDoubleTensor_new(state);

        int weight_shape[4] = {weight->size[0],weight->size[1],weight->size[2],weight->size[3]};
        int out_shape[3] = {output_n->size[0],output_n->size[1],output_n->size[2]};
        int col_shape[2] = {columns->size[0], columns->size[1]};

        THCudaDoubleTensor_resize4d(state, output_n, group, out_shape[0]/group, out_shape[1], out_shape[2]);
        THCudaDoubleTensor_resize5d(state, weight, group, weight_shape[0]/group, weight_shape[1], weight_shape[2], weight_shape[3]);
        THCudaDoubleTensor_resize3d(state, columns, group, col_shape[0]/group, col_shape[1]);

        for(int j=0;j<group;j++){
            THCudaDoubleTensor_select(state, columns_g, columns, 0, j);
            THCudaDoubleTensor_select(state, weight_g, weight, 0, j);
            THCudaDoubleTensor_select(state, output_n_g, output_n, 0, j);

            //GEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
            //C := alpha*op( A )*op( B ) + beta*C,
            int m = weight_g->size[0];
            int k = columns_g->size[0];
            int n = columns_g->size[1];

            THCudaBlas_Dgemm(state, 'n', 'n', n, m, k,
                             1.0f, THCudaDoubleTensor_data(state, columns_g), n,
                             THCudaDoubleTensor_data(state, weight_g), k,
                             0.0f, THCudaDoubleTensor_data(state, output_n_g), n);
        }
        THCudaDoubleTensor_resize3d(state, output_n, out_shape[0], out_shape[1], out_shape[2]);
        THCudaDoubleTensor_resize4d(state, weight, weight_shape[0],weight_shape[1],weight_shape[2],weight_shape[3]);
        THCudaDoubleTensor_resize2d(state, columns, col_shape[0], col_shape[1]);

        THCudaDoubleTensor_free(state, columns_g);
        THCudaDoubleTensor_free(state, weight_g);
        THCudaDoubleTensor_free(state, output_n_g);

    }
    
    THCudaDoubleTensor_free(state, input_n);
    THCudaDoubleTensor_free(state, offset_n);
    THCudaDoubleTensor_free(state, output_n);
    
    return 1;
}


int deform_conv_backward_input_offset_cuda(
        THCudaDoubleTensor *input, THCudaDoubleTensor *weight, THCudaDoubleTensor *offset, THCudaDoubleTensor *grad_output,
        THCudaDoubleTensor *columns, THCudaDoubleTensor *grad_input, THCudaDoubleTensor *grad_offset,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        const int channel_per_deformable_group, const int group) {
    shape_check(state, input, weight, grad_offset, grad_output, columns,
                pad_h, pad_w, stride_h, stride_w, channel_per_deformable_group, group);
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

        THCudaDoubleTensor *columns_g = THCudaDoubleTensor_new(state);
        THCudaDoubleTensor *weight_g = THCudaDoubleTensor_new(state);
        THCudaDoubleTensor *grad_output_n_g = THCudaDoubleTensor_new(state);

        int weight_shape[4] = {weight->size[0],weight->size[1],weight->size[2],weight->size[3]};
        int out_shape[3] = {grad_output_n->size[0],grad_output_n->size[1],grad_output_n->size[2]};
        int col_shape[2] = {columns->size[0], columns->size[1]};

        THCudaDoubleTensor_resize4d(state, grad_output_n, group, out_shape[0]/group, out_shape[1], out_shape[2]);
        THCudaDoubleTensor_resize5d(state, weight, group, weight_shape[0]/group, weight_shape[1], weight_shape[2], weight_shape[3]);
        THCudaDoubleTensor_resize3d(state, columns, group, col_shape[0]/group, col_shape[1]);

        for(int j=0;j<group;j++){
            THCudaDoubleTensor_select(state, columns_g, columns, 0, j);
            THCudaDoubleTensor_select(state, weight_g, weight, 0, j);
            THCudaDoubleTensor_select(state, grad_output_n_g, grad_output_n, 0, j);

            //Wt * O = C
            long m = columns_g->size[0];
            long n = columns_g->size[1];
            long k = weight_g->size[0];

            THCudaBlas_Dgemm(state, 'n', 't', n, m, k,
                             1.0f, THCudaDoubleTensor_data(state, grad_output_n_g), n,
                             THCudaDoubleTensor_data(state, weight_g), m,
                             0.0f, THCudaDoubleTensor_data(state, columns_g), n);
        }

        THCudaDoubleTensor_resize3d(state, grad_output_n, out_shape[0], out_shape[1], out_shape[2]);
        THCudaDoubleTensor_resize4d(state, weight, weight_shape[0],weight_shape[1],weight_shape[2],weight_shape[3]);
        THCudaDoubleTensor_resize2d(state, columns, col_shape[0], col_shape[1]);

        THCudaDoubleTensor_free(state, columns_g);
        THCudaDoubleTensor_free(state, weight_g);
        THCudaDoubleTensor_free(state, grad_output_n_g);

        deformable_col2im_offset(THCState_getCurrentStream(state), THCudaDoubleTensor_data(state, columns),
                                 THCudaDoubleTensor_data(state, input_n), THCudaDoubleTensor_data(state, offset_n),
                                 input->size[1], input->size[2], input->size[3],
                                 grad_output->size[2], grad_output->size[3],
                                 weight->size[2], weight->size[3],
                                 pad_h, pad_w,
                                 stride_h, stride_w,
                                 dilation_h, dilation_w,
                                 channel_per_deformable_group, THCudaDoubleTensor_data(state, grad_offset_n));
        deformable_col2im_input(THCState_getCurrentStream(state),
                                THCudaDoubleTensor_data(state, columns),
                                THCudaDoubleTensor_data(state, offset_n),
                                grad_input->size[1], grad_input->size[2], grad_input->size[3],
                                grad_output->size[2], grad_output->size[3],
                                weight->size[2], weight->size[3],
                                pad_h, pad_w,
                                stride_h, stride_w,
                                dilation_h, dilation_w,
                                channel_per_deformable_group,
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
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        const int channel_per_deformable_group, const int group) {
    shape_check(state, input, grad_weight, offset, grad_output, columns,
                pad_h, pad_w, stride_h, stride_w, channel_per_deformable_group, group);
    THCudaDoubleTensor *input_n = THCudaDoubleTensor_new(state);
    THCudaDoubleTensor *offset_n = THCudaDoubleTensor_new(state);
    THCudaDoubleTensor *grad_output_n = THCudaDoubleTensor_new(state);
    
    for(int i=0;i<input->size[0];i++){
        THCudaDoubleTensor_select(state, input_n, input, 0, i);
        THCudaDoubleTensor_select(state, offset_n, offset, 0, i);
        THCudaDoubleTensor_select(state, grad_output_n, grad_output, 0, i);
            
        deformable_im2col(THCState_getCurrentStream(state),
                          THCudaDoubleTensor_data(state, input_n), THCudaDoubleTensor_data(state, offset_n),
                          input->size[1], input->size[2], input->size[3],
                          grad_output->size[2], grad_output->size[3],
                          grad_weight->size[2], grad_weight->size[3],
                          pad_h, pad_w,
                          stride_h, stride_w,
                          dilation_h, dilation_w,
                          channel_per_deformable_group,
                          THCudaDoubleTensor_data(state, columns));

        THCudaDoubleTensor *columns_g = THCudaDoubleTensor_new(state);
        THCudaDoubleTensor *grad_weight_g = THCudaDoubleTensor_new(state);
        THCudaDoubleTensor *grad_output_n_g = THCudaDoubleTensor_new(state);

        int weight_shape[4] = {grad_weight->size[0],grad_weight->size[1],grad_weight->size[2],grad_weight->size[3]};
        int out_shape[3] = {grad_output_n->size[0],grad_output_n->size[1],grad_output_n->size[2]};
        int col_shape[2] = {columns->size[0], columns->size[1]};

        THCudaDoubleTensor_resize4d(state, grad_output_n, group, out_shape[0]/group, out_shape[1], out_shape[2]);
        THCudaDoubleTensor_resize5d(state, grad_weight, group, weight_shape[0]/group, weight_shape[1], weight_shape[2], weight_shape[3]);
        THCudaDoubleTensor_resize3d(state, columns, group, col_shape[0]/group, col_shape[1]);

        for(int j=0;j<group;j++){
            THCudaDoubleTensor_select(state, columns_g, columns, 0, j);
            THCudaDoubleTensor_select(state, grad_weight_g, grad_weight, 0, j);
            THCudaDoubleTensor_select(state, grad_output_n_g, grad_output_n, 0, j);

            //GEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
            //C := alpha*op( A )*op( B ) + beta*C,
            int m = grad_weight_g->size[0];
            int k = columns_g->size[1];
            int n = columns_g->size[0];
            THCudaBlas_Dgemm(state, 't', 'n', n, m, k,
                             1.0f, THCudaDoubleTensor_data(state, columns_g), k,
                             THCudaDoubleTensor_data(state, grad_output_n_g), k,
                             1.0f, THCudaDoubleTensor_data(state, grad_weight_g), n);

        }

        THCudaDoubleTensor_resize3d(state, grad_output_n, out_shape[0], out_shape[1], out_shape[2]);
        THCudaDoubleTensor_resize4d(state, grad_weight, weight_shape[0],weight_shape[1],weight_shape[2],weight_shape[3]);
        THCudaDoubleTensor_resize2d(state, columns, col_shape[0], col_shape[1]);

        THCudaDoubleTensor_free(state, columns_g);
        THCudaDoubleTensor_free(state, grad_weight_g);
        THCudaDoubleTensor_free(state, grad_output_n_g);

    }
    
    THCudaDoubleTensor_free(state, input_n);
    THCudaDoubleTensor_free(state, offset_n);
    THCudaDoubleTensor_free(state, grad_output_n);

    return 1;
}
