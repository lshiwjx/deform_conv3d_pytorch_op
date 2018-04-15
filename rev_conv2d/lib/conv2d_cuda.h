//
// Created by lshi on 17-9-25.
//

//#ifndef DEFORM3D_NEW_PYTORCH_DEFORM_CONV3D_CUDA_H
//#define DEFORM3D_NEW_PYTORCH_DEFORM_CONV3D_CUDA_H

//#include "TH/TH.h"
//#include "THC/THC.h"
//#include "conv3d_cuda_kernel.h"

//#define THCudaTensor THCudaDoubleTensor
//#define TH_GEMM THCudaBlas_Dgemm
//#define THCudaTensor THCudaTensor
//#define TH_GEMM THCudaBlas_Sgemm

int conv_forward_cuda(
        THCudaTensor *input, THCudaTensor *weight,
        THCudaTensor *columns, THCudaTensor *output,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        const int group);

int conv_backward_input_cuda(
        THCudaTensor *weight, THCudaTensor *grad_output,
        THCudaTensor *columns, THCudaTensor *grad_input,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        const int group);

int conv_backward_weight_cuda(
        THCudaTensor *input, THCudaTensor *grad_output,
        THCudaTensor *columns, THCudaTensor *grad_weight,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        const int group);


//#endif //DEFORM3D_NEW_PYTORCH_DEFORM_CONV3D_CUDA_H
