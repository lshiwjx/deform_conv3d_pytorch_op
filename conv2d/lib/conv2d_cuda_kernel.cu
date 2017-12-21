//
// Created by lshi on 17-9-25.
//

#include "conv2d_cuda_kernel.h"
#include "THC/THC.h"
#include "TH/TH.h"

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old); }
#endif

#define THREAD_PRE_BLOCK 1024

//--------------------------------------------------forward---------------------------------------------
template<typename DType>
__global__ void im2col_gpu_kernel(
        const int num_kernels, const DType *data_im,
        const int input_c, const int input_h, const int input_w,
        const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        const int output_h, const int output_w,
        DType *data_col) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < num_kernels;
         index += blockDim.x * gridDim.x) {
        const int input_v = input_w * input_h;
        const int output_v = output_h * output_w;
        const int kernel_v = kernel_h * kernel_w;
        //CH'W'H"W"
        const int w_out = index % output_w;
        const int h_out = index / output_w % output_h;
        const int w_kernel = index / output_v % kernel_w;
        const int h_kernel = index / output_v / kernel_w % kernel_h;
        const int c_in = index / output_v / kernel_v % input_c;

        const int h_in = h_out * stride_h - pad_h;
        const int w_in = w_out * stride_w - pad_w;

        //(CH'W')(H"W")
        DType *data_col_base_ptr = data_col +
                                   (c_in * kernel_v +
                                    h_kernel * kernel_w +
                                    w_kernel) * output_v +
                                   h_out * output_w +
                                   w_out;
        //CHW
        const DType *data_in_base_ptr = data_im + c_in * input_v;

        const int h_in_after = h_in + h_kernel*dilation_h;
        const int w_in_after = w_in + w_kernel*dilation_w;

        DType val = 0;
        if (h_in_after > -1 && w_in_after > -1 &&
            h_in_after < input_h && w_in_after < input_w) {
            val = data_in_base_ptr[h_in_after * input_w + w_in_after];

        }
        *data_col_base_ptr = val;
    }
}


inline int get_cuda_blocks(const int num_kernel) {
    return (num_kernel + THREAD_PRE_BLOCK - 1) / THREAD_PRE_BLOCK;
}

void im2col(cudaStream_t stream,
           const float *data_in,
           const int input_c,
           const int input_h, const int input_w,
           const int output_h, const int output_w,
           const int kernel_h, const int kernel_w,
           const int pad_h, const int pad_w,
           const int stride_h, const int stride_w,
           const int dilation_h, const int dilation_w,
           float *data_col) {
    int num_kernels = output_h * output_w * input_c * kernel_h * kernel_w;
    im2col_gpu_kernel << < get_cuda_blocks(num_kernels), THREAD_PRE_BLOCK, 0, stream >> > (
            num_kernels, data_in,
                    input_c, input_h, input_w,
                    kernel_h, kernel_w,
                    pad_h, pad_w,
                    stride_h, stride_w,
                    dilation_h, dilation_w,
                    output_h, output_w,
                    data_col);
}

//---------------------------------------------backward to input---------------------------------------------------
template<typename DType>
__global__ void col2im_gpu_kernel(
        const int num_kernels, const DType *data_col,
        const int input_c,
        const int input_h, const int input_w,
        const int output_h, const int output_w,
        const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int dilation_h, const int dilation_w,
        DType *grad_im) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < num_kernels;
         index += blockDim.x * gridDim.x) {
        const int input_v = input_w * input_h;
        const int output_v = output_h * output_w;
        const int kernel_v = kernel_h * kernel_w;

        //H"W"CH'W'
        const int w_kernel = index % kernel_w;
        const int h_kernel = index / kernel_w % kernel_h;
        const int c_in = index / kernel_v % input_c;
        const int w_out = index / kernel_v / input_c % output_w;
        const int h_out = index / kernel_v / input_c / output_w % output_h;

        const int h_in = h_out * stride_h - pad_h;
        const int w_in = w_out * stride_w - pad_w;

        //CH'W' H"W"
        const DType *data_col_base_ptr = data_col +
                                         (c_in * kernel_v +
                                          h_kernel * kernel_w +
                                          w_kernel) * output_v +
                                          h_out * output_w +
                                          w_out;

        //CHW
        DType *grad_in_base_ptr = grad_im + c_in * input_v;
        const int width = input_w;
        const int h_in_after = h_in + h_kernel*dilation_h;
        const int w_in_after = w_in + w_kernel*dilation_w;
        if (h_in_after > -1 && w_in_after > -1 &&
            h_in_after < input_h && w_in_after < input_w) {
            atomicAdd(grad_in_base_ptr + h_in_after * width + w_in_after, *data_col_base_ptr);
        }
    }
}

void col2im(cudaStream_t stream,
         const float *data_col,
         const int input_c,
         const int input_h, const int input_w,
         const int output_h, const int output_w,
         const int kernel_h, const int kernel_w,
         const int pad_h, const int pad_w,
         const int stride_h, const int stride_w,
         const int dilation_h, const int dilation_w,
         float *grad_im) {
    const int num_kernels = output_h * output_w * input_c * kernel_h * kernel_w;
    col2im_gpu_kernel << < get_cuda_blocks(num_kernels), THREAD_PRE_BLOCK, 0, stream >> > (
            num_kernels, data_col,
                    input_c, input_h, input_w,
                    output_h, output_w,
                    kernel_h, kernel_w,
                    pad_h, pad_w,
                    stride_h, stride_w,
                    dilation_h, dilation_w,
                    grad_im
    );
}