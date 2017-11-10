//
// Created by lshi on 17-9-25.
//

#include "deform_conv2d_cuda_kernel.h"
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
__device__ DType Bi_Linear(const DType *bottom_data,
                           const int height, const int width,
                           const double h, const double w) {

    //get the cube, the function floor can not be used in template
    int h_low = int(h);
    int w_low = int(w);
    int h_high = (h >= height - 1 || h <= 0) ? h_low : h_low + 1;
    int w_high = (w >= width - 1 || w <= 0) ? w_low : w_low + 1;

    //the corner, format is hw
    DType c00 = bottom_data[h_low * width + w_low];
    DType c01 = bottom_data[h_low * width + w_high];
    DType c10 = bottom_data[h_high * width + w_low];
    DType c11 = bottom_data[h_high * width + w_high];

    //calculate the distance between the point and corner, using 1 to make sure using the low if equal
    DType l_width = w - w_low;
    DType h_width = 1 - l_width;
    DType l_height = h - h_low;
    DType h_height = 1 - l_height;


    //interpolation
    DType c0 = c00 * h_width + c01 * l_width;
    DType c1 = c10 * h_width + c11 * l_width;

    DType c = c0 * h_height + c1 * l_height;

    return c;
}


template<typename DType>
__global__ void deformable_im2col_gpu_kernel(
        const int num_kernels, const DType *data_im, const DType *data_offset,
        const int input_c, const int input_h, const int input_w,
        const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int channel_per_deformable_group,
        const int output_h, const int output_w, DType *data_col) {
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
        const int g_off = c_in / channel_per_deformable_group;
//        const int deform_group = input_c / channel_per_deformable_group;
//        if (threadIdx.x == 0)
//        printf("%d %d %d %d %d %d\n",threadIdx.x,c_in,h_kernel,w_kernel,h_out,w_out);
//        printf("%d\n",index);
        //(CH'W')(H"W")
        DType *data_col_base_ptr = data_col +
                                   (c_in * kernel_v +
                                    h_kernel * kernel_w +
                                    w_kernel) * output_v +
                                   h_out * output_w +
                                   w_out;
        //CHW
        const DType *data_in_base_ptr = data_im + c_in * input_v;
        //GH'W'2H"W"
        const DType *data_offset_base_ptr = data_offset +
                                            (g_off * kernel_v +
                                             h_kernel * kernel_w +
                                             w_kernel) * 2 * output_v;

//        printf("%d %f %f %f\n",threadIdx.x, *data_col_base_ptr, *data_in_base_ptr, *data_offset_base_ptr);

        const int offset = h_out * output_w + w_out;
//        printf("%d %d %d %d %f %f %f\n",threadIdx.x,l_in,h_in,w_in,data_offset_base_ptr[offset + output_v * 0],
//               data_offset_base_ptr[offset + output_v * 1],data_offset_base_ptr[offset + output_v * 2]);
        const DType h_in_after = h_in + h_kernel + data_offset_base_ptr[offset + output_v * 0];
        const DType w_in_after = w_in + w_kernel + data_offset_base_ptr[offset + output_v * 1];
//        printf("%d %f %f %f\n",threadIdx.x,l_in_after,h_in_after,w_in_after);

        DType val = 0;
        if (h_in_after > -1 && w_in_after > -1 &&
            h_in_after < input_h && w_in_after < input_w) {
            //interpolation
            val = Bi_Linear(data_in_base_ptr, input_h, input_w,
                            h_in_after, w_in_after);
        }
        *data_col_base_ptr = val;
//        if (threadIdx.x==0)
//        printf("%d %f %f %f\n",threadIdx.x,h_in_after,w_in_after ,val);
    }
}


inline int get_cuda_blocks(const int num_kernel) {
    return (num_kernel + THREAD_PRE_BLOCK - 1) / THREAD_PRE_BLOCK;
}

void deformable_im2col(cudaStream_t stream,
                       const double *data_in, const double *data_offset,
                       const int input_c,
                       const int input_h, const int input_w,
                       const int output_h, const int output_w,
                       const int kernel_h, const int kernel_w,
                       const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int channel_per_deformable_group, double *data_col) {
    int num_kernels = output_h * output_w * input_c * kernel_h * kernel_w;
    deformable_im2col_gpu_kernel << < get_cuda_blocks(num_kernels), THREAD_PRE_BLOCK, 0, stream >> > (
            num_kernels, data_in, data_offset,
                    input_c, input_h, input_w,
                    kernel_h, kernel_w,
                    pad_h, pad_w,
                    stride_h, stride_w,
                    channel_per_deformable_group,
                    output_h, output_w,
                    data_col);
}

//---------------------------------------------backward to input---------------------------------------------------
template<typename DType>
__global__ void deformable_col2im_input_gpu_kernel(
        const int num_kernels, const DType *data_col, const DType *data_offset,
        const int input_c,
        const int input_h, const int input_w,
        const int output_h, const int output_w,
        const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int channel_per_deformable_group, DType *grad_im) {
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
        const int g_off = c_in / channel_per_deformable_group;
//        const int deform_group = input_c / channel_per_deformable_group;

        //CH'W' H"W"
        const DType *data_col_base_ptr = data_col +
                                         (c_in * kernel_v +
                                          h_kernel * kernel_w +
                                          w_kernel) * output_v +
                                         h_out * output_w +
                                         w_out;

        //GH'W'3H"W"
        int offset_base = (g_off * kernel_v +
                           h_kernel * kernel_w +
                           w_kernel) * output_v * 2;
        int offset = h_out * output_w + w_out;
        const DType *data_offset_base_ptr = data_offset + offset_base;
        //CHW
        DType *grad_in_base_ptr = grad_im + c_in * input_v;
        const int width = input_w;
        const DType h_in_after = h_in + h_kernel + data_offset_base_ptr[0 * output_v + offset];
        const DType w_in_after = w_in + w_kernel + data_offset_base_ptr[1 * output_v + offset];
        if (h_in_after > -1 && w_in_after > -1 &&
            h_in_after < input_h && w_in_after < input_w) {
            //eight point around
            int h_low = int(h_in_after);
            int w_low = int(w_in_after);

            int h_high = (h_in_after >= input_h - 1 || h_in_after <= 0) ? h_low : h_low + 1;
            int w_high = (w_in_after >= input_w - 1 || w_in_after <= 0) ? w_low : w_low + 1;

            int a00 = h_low * width + w_low;
            int a01 = h_low * width + w_high;
            int a10 = h_high * width + w_low;
            int a11 = h_high * width + w_high;

            DType l_width = w_in_after - w_low;
            DType h_width = 1 - l_width;
            DType l_height = h_in_after - h_low;
            DType h_height = 1 - l_height;

            //grad for input
            atomicAdd(
                    grad_in_base_ptr + a00,
                    h_height * h_width *
                    (*data_col_base_ptr));
            atomicAdd(
                    grad_in_base_ptr + a01,
                    h_height * l_width *
                    (*data_col_base_ptr));
            atomicAdd(
                    grad_in_base_ptr + a10,
                    l_height * h_width *
                    (*data_col_base_ptr));
            atomicAdd(
                    grad_in_base_ptr + a11,
                    l_height * l_width *
                    (*data_col_base_ptr));
        }
    }
}


void deformable_col2im_input(cudaStream_t stream,
                             const double *data_col, const double *data_offset,
                             const int input_c,
                             const int input_h, const int input_w,
                             const int output_h, const int output_w,
                             const int kernel_h, const int kernel_w,
                             const int pad_h, const int pad_w,
                             const int stride_h, const int stride_w,
                             const int channel_per_deformable_group, double *grad_im) {
    const int num_kernels = output_h * output_w * input_c * kernel_h * kernel_w;
    deformable_col2im_input_gpu_kernel << < get_cuda_blocks(num_kernels), THREAD_PRE_BLOCK, 0, stream >> > (
            num_kernels, data_col, data_offset,
                    input_c, input_h, input_w,
                    output_h, output_w,
                    kernel_h, kernel_w,
                    pad_h, pad_w,
                    stride_h, stride_w,
                    channel_per_deformable_group, grad_im
    );
}

//--------------------------------------------------backward to offset---------------------------------------------


template<typename DType>
__global__ void deformable_col2im_offset_gpu_kernel(
        const int num_kernels, const DType *data_col, const DType *data_im, const DType *data_offset,
        const int input_c,
        const int input_h, const int input_w,
        const int output_h, const int output_w,
        const int kernel_h, const int kernel_w,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int channel_per_deformable_group,
        DType *grad_off) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < num_kernels;
         index += blockDim.x * gridDim.x) {
        //GH'W'2H"W"
        const int input_v = input_w * input_h;
        const int output_v = output_h * output_w;
        const int kernel_v = kernel_h * kernel_w;

        const int deform_group = input_c / channel_per_deformable_group;

        const int w_out = index % output_w;
        const int h_out = index / output_w % output_h;
        const int int_2 = index / output_v % 2;
        const int w_kernel = index / output_v / 2 % kernel_w;
        const int h_kernel = index / output_v / 2 / kernel_w % kernel_h;
        const int g_off = index / output_v / 2 / kernel_v % deform_group;

        const int h_in = h_out * stride_h - pad_h;
        const int w_in = w_out * stride_w - pad_w;

        //GH"W"H'W'2
        int offset_base = (g_off * kernel_v +
                           h_kernel * kernel_w +
                           w_kernel) * output_v * 2;
        int offset = h_out * output_w + w_out;
        const DType *data_offset_base_ptr = data_offset + offset_base;
        DType *grad_offset_base_ptr = grad_off + offset_base + int_2 * output_v + offset;
        DType val = 0;
        for (int i = 0; i < channel_per_deformable_group; ++i) {
            const int c_in = g_off * channel_per_deformable_group + i;
            //CH'W' H"W"
            const DType *data_col_base_ptr = data_col +
                                             (c_in * kernel_v +
                                              h_kernel * kernel_w +
                                              w_kernel) * output_v +
                                             h_out * output_w +
                                             w_out;
            //CHW
            const DType *data_in_base_ptr = data_im + c_in * input_v;

            const int width = input_w;
            const DType h_in_after = h_in + h_kernel + data_offset_base_ptr[0 * output_v + offset];
            const DType w_in_after = w_in + w_kernel + data_offset_base_ptr[1 * output_v + offset];
            if (h_in_after > -1 && w_in_after > -1 &&
                h_in_after < input_h && w_in_after < input_w) {
                int h_low = int(h_in_after);
                int w_low = int(w_in_after);

                int h_high = (h_in_after >= input_h - 1 || h_in_after <= 0) ? h_low : h_low + 1;
                int w_high = (w_in_after >= input_w - 1 || w_in_after <= 0) ? w_low : w_low + 1;

                int a00 = h_low * width + w_low;
                int a01 = h_low * width + w_high;
                int a10 = h_high * width + w_low;
                int a11 = h_high * width + w_high;

                //value of eight point
                DType c00 = data_in_base_ptr[a00];
                DType c01 = data_in_base_ptr[a01];
                DType c10 = data_in_base_ptr[a10];
                DType c11 = data_in_base_ptr[a11];

                //six distance
                DType l_width = w_in_after - w_low;
                DType h_width = 1 - l_width;
                DType l_height = h_in_after - h_low;
                DType h_height = 1 - l_height;

                //h:1+ w:0*h_width
                switch (int_2) {
                    case 0:
                        val += *data_col_base_ptr *
                               (c11 * l_width + c10 * h_width -
                                c01 * l_width - c00 * h_width);
                        break;
                    case 1:
                        val += *data_col_base_ptr *
                               (c01 * h_height + c11 * l_height -
                                c00 * h_height - c10 * l_height);
                        break;
                    default:
                        printf("error in switch");
                }
            }
        }
        *grad_offset_base_ptr = val;
    }
}


void deformable_col2im_offset(cudaStream_t stream,
                              const double *data_col, const double *data_im, const double *data_offset,
                              const int input_c, const int input_h, const int input_w,
                              const int output_h, const int output_w,
                              const int kernel_h, const int kernel_w,
                              const int pad_h, const int pad_w,
                              const int stride_h, const int stride_w,
                              const int channel_per_deformable_group,
                              double *grad_offset) {
    const int num_kernels = (input_c / channel_per_deformable_group)
                            * kernel_h * kernel_w * 2 * output_h * output_w;
    deformable_col2im_offset_gpu_kernel << < get_cuda_blocks(num_kernels), THREAD_PRE_BLOCK, 0, stream >> > (
            num_kernels, data_col, data_im, data_offset,
                    input_c, input_h, input_w,
                    output_h, output_w,
                    kernel_h, kernel_w,
                    pad_h, pad_w,
                    stride_h, stride_w,
                    channel_per_deformable_group, grad_offset
    );
}
