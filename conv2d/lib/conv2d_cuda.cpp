//
// Created by lshi on 17-9-25.
//

//#include "conv2d_cuda.h"
#include <THC/THCStorage.h>
#include "TH/TH.h"
#include "THC/THC.h"
#include "conv2d_cuda_kernel.h"
#include "iostream"

using namespace std;
extern THCState *state;

//#define THCudaTensor THCudaDoubleTensor
//#define THCudaBlas_Sgemm THCudaBlas_Dgemm
//#define THCudaTensor THCudaTensor
//#define THCudaBlas_Sgemm THCudaBlas_Sgemm

inline int input_to_output(int input, int pad, int dilation, int kernel, int stride) {
    return (input + 2 * pad - dilation*(kernel - 1) - 1) / stride + 1;
}

void check(bool e, const char *msg) {
    if (!e)
        THError(msg);
}

void shape_check(THCState *state,
                 THCudaTensor *input, THCudaTensor *weight, THCudaTensor *output,
                 THCudaTensor *columns,
                 const int pad_h, const int pad_w,
                 const int stride_h, const int stride_w,
                 const int dilation_h, const int dilation_w,
                 const int group) {
    THCAssertSameGPU(THCudaTensor_checkGPU(state, 4, input, weight, output, columns));

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

    int columns_dim = columns->nDimension;
    check(columns_dim == 2, "columns dim");
    long columns_row = columns->size[0];
    check(columns_row == input_c * kernel_h * kernel_w, "columns row");
    long columns_col = columns->size[1];
    check(columns_col == output_h * output_w, "columns col");

    long output_hh = input_to_output(input_h, pad_h, dilation_h, kernel_h, stride_h);
    check(output_hh == output_h, "error outhh");
    long output_ww = input_to_output(input_w, pad_w, dilation_w, kernel_w, stride_w);
    check(output_ww == output_w, "error outww");
}


int conv_forward_cuda(
        THCudaTensor *input, THCudaTensor *weight,
        THCudaTensor *columns, THCudaTensor *output,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        const int group) {
    shape_check(state, input, weight, output, columns,
                pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, group);
    
    THCudaTensor *input_n = THCudaTensor_new(state);
    THCudaTensor *output_n = THCudaTensor_new(state);
    
    for(int i=0;i<input->size[0];i++){
        THCudaTensor_select(state, input_n, input, 0, i);
        THCudaTensor_select(state, output_n, output, 0, i);
        
        im2col(THCState_getCurrentStream(state),
                  THCudaTensor_data(state, input_n),
                  input->size[1], input->size[2], input->size[3],
                  output->size[2], output->size[3],
                  weight->size[2], weight->size[3],
                  pad_h, pad_w,
                  stride_h, stride_w,
                  dilation_h, dilation_w,
                  THCudaTensor_data(state, columns));

        THCudaTensor *columns_g = THCudaTensor_new(state);
        THCudaTensor *weight_g = THCudaTensor_new(state);
        THCudaTensor *output_n_g = THCudaTensor_new(state);

        int weight_shape[4] = {weight->size[0],weight->size[1],weight->size[2],weight->size[3]};
        int out_shape[3] = {output_n->size[0],output_n->size[1],output_n->size[2]};
        int col_shape[2] = {columns->size[0], columns->size[1]};

        THCudaTensor_resize4d(state, output_n, group, out_shape[0]/group, out_shape[1], out_shape[2]);
        THCudaTensor_resize5d(state, weight, group, weight_shape[0]/group, weight_shape[1], weight_shape[2], weight_shape[3]);
        THCudaTensor_resize3d(state, columns, group, col_shape[0]/group, col_shape[1]);

        for(int j=0;j<group;j++){
            THCudaTensor_select(state, columns_g, columns, 0, j);
            THCudaTensor_select(state, weight_g, weight, 0, j);
            THCudaTensor_select(state, output_n_g, output_n, 0, j);

            //GEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
            //C := alpha*op( A )*op( B ) + beta*C,
            int m = weight_g->size[0];
            int k = columns_g->size[0];
            int n = columns_g->size[1];

            THCudaBlas_Sgemm(state, 'n', 'n', n, m, k,
                             1.0f, THCudaTensor_data(state, columns_g), n,
                             THCudaTensor_data(state, weight_g), k,
                             0.0f, THCudaTensor_data(state, output_n_g), n);
        }
        THCudaTensor_resize3d(state, output_n, out_shape[0], out_shape[1], out_shape[2]);
        THCudaTensor_resize4d(state, weight, weight_shape[0],weight_shape[1],weight_shape[2],weight_shape[3]);
        THCudaTensor_resize2d(state, columns, col_shape[0], col_shape[1]);

        THCudaTensor_free(state, columns_g);
        THCudaTensor_free(state, weight_g);
        THCudaTensor_free(state, output_n_g);

    }
    
    THCudaTensor_free(state, input_n);
    THCudaTensor_free(state, output_n);
    
    return 1;
}


int conv_backward_input_cuda(
        THCudaTensor *weight, THCudaTensor *grad_output,
        THCudaTensor *columns, THCudaTensor *grad_input,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        const int group) {
    shape_check(state, grad_input, weight, grad_output, columns,
                pad_h, pad_w, stride_h, stride_w,  dilation_h, dilation_w, group);

    THCudaTensor *grad_input_n = THCudaTensor_new(state);
    THCudaTensor *grad_output_n = THCudaTensor_new(state);
    
    for(int i=0;i<grad_input->size[0];i++){
        THCudaTensor_select(state, grad_input_n, grad_input, 0, i);
        THCudaTensor_select(state, grad_output_n, grad_output, 0, i);

        THCudaTensor *columns_g = THCudaTensor_new(state);
        THCudaTensor *weight_g = THCudaTensor_new(state);
        THCudaTensor *grad_output_n_g = THCudaTensor_new(state);

        int weight_shape[4] = {weight->size[0],weight->size[1],weight->size[2],weight->size[3]};
        int out_shape[3] = {grad_output_n->size[0],grad_output_n->size[1],grad_output_n->size[2]};
        int col_shape[2] = {columns->size[0], columns->size[1]};

        THCudaTensor_resize4d(state, grad_output_n, group, out_shape[0]/group, out_shape[1], out_shape[2]);
        THCudaTensor_resize5d(state, weight, group, weight_shape[0]/group, weight_shape[1], weight_shape[2], weight_shape[3]);
        THCudaTensor_resize3d(state, columns, group, col_shape[0]/group, col_shape[1]);

        for(int j=0;j<group;j++){
            THCudaTensor_select(state, columns_g, columns, 0, j);
            THCudaTensor_select(state, weight_g, weight, 0, j);
            THCudaTensor_select(state, grad_output_n_g, grad_output_n, 0, j);

            //Wt * O = C
            long m = columns_g->size[0];
            long n = columns_g->size[1];
            long k = weight_g->size[0];

            THCudaBlas_Sgemm(state, 'n', 't', n, m, k,
                             1.0f, THCudaTensor_data(state, grad_output_n_g), n,
                             THCudaTensor_data(state, weight_g), m,
                             0.0f, THCudaTensor_data(state, columns_g), n);
        }

        THCudaTensor_resize3d(state, grad_output_n, out_shape[0], out_shape[1], out_shape[2]);
        THCudaTensor_resize4d(state, weight, weight_shape[0],weight_shape[1],weight_shape[2],weight_shape[3]);
        THCudaTensor_resize2d(state, columns, col_shape[0], col_shape[1]);

        THCudaTensor_free(state, columns_g);
        THCudaTensor_free(state, weight_g);
        THCudaTensor_free(state, grad_output_n_g);

        col2im(THCState_getCurrentStream(state),
            THCudaTensor_data(state, columns),
            grad_input->size[1], grad_input->size[2], grad_input->size[3],
            grad_output->size[2], grad_output->size[3],
            weight->size[2], weight->size[3],
            pad_h, pad_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            THCudaTensor_data(state, grad_input_n));
    }
    
    THCudaTensor_free(state, grad_input_n);
    THCudaTensor_free(state, grad_output_n);


    return 1;
}


int conv_backward_weight_cuda(
        THCudaTensor *input, THCudaTensor *grad_output,
        THCudaTensor *columns, THCudaTensor *grad_weight,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        const int group) {
    shape_check(state, input, grad_weight, grad_output, columns,
                pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, group);
    THCudaTensor *input_n = THCudaTensor_new(state);
    THCudaTensor *grad_output_n = THCudaTensor_new(state);
    
    for(int i=0;i<input->size[0];i++){
        THCudaTensor_select(state, input_n, input, 0, i);
        THCudaTensor_select(state, grad_output_n, grad_output, 0, i);
            
        im2col(THCState_getCurrentStream(state),
                          THCudaTensor_data(state, input_n),
                          input->size[1], input->size[2], input->size[3],
                          grad_output->size[2], grad_output->size[3],
                          grad_weight->size[2], grad_weight->size[3],
                          pad_h, pad_w,
                          stride_h, stride_w,
                          dilation_h, dilation_w,
                          THCudaTensor_data(state, columns));

        THCudaTensor *columns_g = THCudaTensor_new(state);
        THCudaTensor *grad_weight_g = THCudaTensor_new(state);
        THCudaTensor *grad_output_n_g = THCudaTensor_new(state);

        int weight_shape[4] = {grad_weight->size[0],grad_weight->size[1],grad_weight->size[2],grad_weight->size[3]};
        int out_shape[3] = {grad_output_n->size[0],grad_output_n->size[1],grad_output_n->size[2]};
        int col_shape[2] = {columns->size[0], columns->size[1]};

        THCudaTensor_resize4d(state, grad_output_n, group, out_shape[0]/group, out_shape[1], out_shape[2]);
        THCudaTensor_resize5d(state, grad_weight, group, weight_shape[0]/group, weight_shape[1], weight_shape[2], weight_shape[3]);
        THCudaTensor_resize3d(state, columns, group, col_shape[0]/group, col_shape[1]);

        for(int j=0;j<group;j++){
            THCudaTensor_select(state, columns_g, columns, 0, j);
            THCudaTensor_select(state, grad_weight_g, grad_weight, 0, j);
            THCudaTensor_select(state, grad_output_n_g, grad_output_n, 0, j);

            //GEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
            //C := alpha*op( A )*op( B ) + beta*C,
            int m = grad_weight_g->size[0];
            int k = columns_g->size[1];
            int n = columns_g->size[0];
            THCudaBlas_Sgemm(state, 't', 'n', n, m, k,
                             1.0f, THCudaTensor_data(state, columns_g), k,
                             THCudaTensor_data(state, grad_output_n_g), k,
                             1.0f, THCudaTensor_data(state, grad_weight_g), n);

        }

        THCudaTensor_resize3d(state, grad_output_n, out_shape[0], out_shape[1], out_shape[2]);
        THCudaTensor_resize4d(state, grad_weight, weight_shape[0],weight_shape[1],weight_shape[2],weight_shape[3]);
        THCudaTensor_resize2d(state, columns, col_shape[0], col_shape[1]);

        THCudaTensor_free(state, columns_g);
        THCudaTensor_free(state, grad_weight_g);
        THCudaTensor_free(state, grad_output_n_g);

    }
    
    THCudaTensor_free(state, input_n);
    THCudaTensor_free(state, grad_output_n);

    return 1;
}
