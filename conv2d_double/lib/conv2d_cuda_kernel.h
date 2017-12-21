//
// Created by lshi on 17-9-25.
//

//#ifndef DEFORM3D_NEW_PYTORCH_DEFORM_CONV3D_CUDA_BACKWARD_CU_H
//#define DEFORM3D_NEW_PYTORCH_DEFORM_CONV3D_CUDA_BACKWARD_CU_H

//#include "THC/THC.h"
//#include "TH/TH.h"

void im2col(cudaStream_t stream,
           const double *data_in,
           const int input_c,
           const int input_h, const int input_w,
           const int output_h, const int output_w,
           const int kernel_h, const int kernel_w,
           const int pad_h, const int pad_w,
           const int stride_h, const int stride_w,
           const int dilation_h, const int dilation_w,
           double *data_col);


void col2im(cudaStream_t stream,
         const double *data_col,
         const int input_c,
         const int input_h, const int input_w,
         const int output_h, const int output_w,
         const int kernel_h, const int kernel_w,
         const int pad_h, const int pad_w,
         const int stride_h, const int stride_w,
         const int dilation_h, const int dilation_w,
         double *grad_im);
//#endif //DEFORM3D_NEW_PYTORCH_DEFORM_CONV3D_CUDA_BACKWARD_CU_H
