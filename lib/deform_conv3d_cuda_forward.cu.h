//
// Created by lshi on 17-9-25.
//

#ifndef DEFORM3D_NEW_PYTORCH_DEFORM_CONV_CUDA_H
#define DEFORM3D_NEW_PYTORCH_DEFORM_CONV_CUDA_H

#include "TH/TH.h"
#include "THC/THC.h"

inline int get_cuda_blocks(const int num_kernel);

template<typename DType>
void deformable_im2col(cudaStream_t stream,
                       const DType *data_in, const DType *data_offset,
                       const int batch_size, const int input_c,
                       const int input_l, const int input_h, const int input_w,
                       const int output_l, const int output_h, const int output_w,
                       const int kernel_l, const int kernel_h, const int kernel_w,
                       const int pad_l, const int pad_h, const int pad_w,
                       const int stride_l, const int stride_h, const int stride_w,
                       const int channel_per_deformable_group, DType *data_col);

#endif //DEFORM3D_NEW_PYTORCH_DEFORM_CONV_CUDA_H
