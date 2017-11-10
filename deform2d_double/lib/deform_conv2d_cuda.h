//
// Created by lshi on 17-9-25.
//

//#ifndef DEFORM3D_NEW_PYTORCH_DEFORM_CONV3D_CUDA_H
//#define DEFORM3D_NEW_PYTORCH_DEFORM_CONV3D_CUDA_H

//#include "TH/TH.h"
//#include "THC/THC.h"
//#include "deform_conv3d_cuda_kernel.h"

int deform_conv_forward_cuda(
        THCudaDoubleTensor *input, THCudaDoubleTensor *weight, THCudaDoubleTensor *offset,
        THCudaDoubleTensor *columns, THCudaDoubleTensor *output,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int channel_per_deformable_group);

int deform_conv_backward_input_offset_cuda(
        THCudaDoubleTensor *input, THCudaDoubleTensor *weight, THCudaDoubleTensor *offset, THCudaDoubleTensor *grad_output,
        THCudaDoubleTensor *columns, THCudaDoubleTensor *grad_input, THCudaDoubleTensor *grad_offset,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int channel_per_deformable_group);

int deform_conv_backward_weight_cuda(
        THCudaDoubleTensor *input, THCudaDoubleTensor *offset, THCudaDoubleTensor *grad_output,
        THCudaDoubleTensor *columns, THCudaDoubleTensor *grad_weight,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int channel_per_deformable_group);


//#endif //DEFORM3D_NEW_PYTORCH_DEFORM_CONV3D_CUDA_H
