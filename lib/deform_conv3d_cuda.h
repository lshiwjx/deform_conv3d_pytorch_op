//
// Created by lshi on 17-9-25.
//

#ifndef DEFORM3D_NEW_PYTORCH_DEFORM_CONV3D_CUDA_H
#define DEFORM3D_NEW_PYTORCH_DEFORM_CONV3D_CUDA_H

#include "TH/TH.h"
#include "THC/THC.h"
#include "deform_conv3d_cuda_backward.cu.h"
#include "deform_conv3d_cuda_forward.cu.h"

int deform_conv_forward_cuda(
        THCudaTensor *input, THCudaTensor *weight, THCudaTensor *offset, THCudaTensor *columns,
        THCudaTensor *output, THCudaTensor *ones,
        const int kernel_l, const int kernel_h, const int kernel_w,
        const int pad_l, const int pad_h, const int pad_w,
        const int stride_l, const int stride_h, const int stride_w,
        const int channel_per_deformable_group);

int deform_conv_backward_input_offset_cuda(
        THCudaTensor *input, THCudaTensor *weight, THCudaTensor *offset, THCudaTensor *columns,
        THCudaTensor *gradOutput, THCudaTensor *gradInput, THCudaTensor *gradOffset,
        const int kernel_l, const int kernel_h, const int kernel_w,
        const int pad_l, const int pad_h, const int pad_w,
        const int stride_l, const int stride_h, const int stride_w,
        const int channel_per_deformable_group);

int deform_conv_backward_weight_cuda(
        THCudaTensor *input, THCudaTensor *offset, THCudaTensor *columns,
        THCudaTensor *gradOutput, THCudaTensor *gradWeight, THCudaTensor *ones,
        const int kernel_l, const int kernel_h, const int kernel_w,
        const int pad_l, const int pad_h, const int pad_w,
        const int stride_l, const int stride_h, const int stride_w,
        const int channel_per_deformable_group);


#endif //DEFORM3D_NEW_PYTORCH_DEFORM_CONV3D_CUDA_H
