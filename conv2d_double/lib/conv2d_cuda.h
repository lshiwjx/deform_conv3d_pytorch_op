//
// Created by lshi on 17-9-25.
//

//#ifndef DEFORM3D_NEW_PYTORCH_DEFORM_CONV3D_CUDA_H
//#define DEFORM3D_NEW_PYTORCH_DEFORM_CONV3D_CUDA_H

//#include "TH/TH.h"
//#include "THC/THC.h"
//#include "conv3d_cuda_kernel.h"

int conv_forward_cuda(
        THCudaDoubleTensor *input, THCudaDoubleTensor *weight,
        THCudaDoubleTensor *columns, THCudaDoubleTensor *output,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        const int group);

int conv_backward_input_cuda(
        THCudaDoubleTensor *weight, THCudaDoubleTensor *grad_output,
        THCudaDoubleTensor *columns, THCudaDoubleTensor *grad_input,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        const int group);

int conv_backward_weight_cuda(
        THCudaDoubleTensor *input, THCudaDoubleTensor *grad_output,
        THCudaDoubleTensor *columns, THCudaDoubleTensor *grad_weight,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        const int group);


//#endif //DEFORM3D_NEW_PYTORCH_DEFORM_CONV3D_CUDA_H
