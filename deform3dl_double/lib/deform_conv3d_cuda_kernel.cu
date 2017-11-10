//
// Created by lshi on 17-9-25.
//

#include "deform_conv3d_cuda_kernel.h"
#include "THC/THC.h"
#include "TH/TH.h"

#define THREAD_PRE_BLOCK 1024
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

//--------------------------------------------------forward---------------------------------------------
template<typename DType>
__device__ DType Tri_Linear(const DType *bottom_data,
                            const int length, const int height, const int width,
                            const double l, const int h, const int w) {
    //length and area
    const int data_width_1d = width;
    const int data_width_2d = height * width;

    //get the cube, the function int can not be used in template
    int l_low = int(l);
    int l_high = (l >= length - 1 || l <= 0) ? l_low : l_low + 1;

    //the corner, format is lhw
    DType c0 = bottom_data[l_low * data_width_2d + h * data_width_1d + w];
    DType c1 = bottom_data[l_high * data_width_2d + h * data_width_1d + w];


    //calculate the distance between the point and corner, using 1 to make sure using the low if equal
    DType l_length = l - l_low;
    DType h_length = 1 - l_length;

    //interpolation
    DType c = c0 * h_length + c1 * l_length;

    return c;
}


template<typename DType>
__global__ void deformable_im2col_gpu_kernel(
        const int num_kernels, const DType *data_im, const DType *data_offset,
        const int input_c, const int input_l, const int input_h, const int input_w,
        const int kernel_l, const int kernel_h, const int kernel_w,
        const int pad_l, const int pad_h, const int pad_w,
        const int stride_l, const int stride_h, const int stride_w,
        const int channel_per_deformable_group,
        const int output_l, const int output_h, const int output_w, DType *data_col) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < num_kernels;
         index += blockDim.x * gridDim.x) {
        const int input_v = input_l * input_w * input_h;
        const int output_v = output_l * output_h * output_w;
        const int kernel_v = kernel_l * kernel_h * kernel_w;
        //L"H"W"CL'H'W'
        const int w_out = index % output_w;
        const int h_out = index / output_w % output_h;
        const int l_out = index / output_w / output_h % output_l;
        const int w_kernel = index / output_v % kernel_w;
        const int h_kernel = index / output_v / kernel_w % kernel_h;
        const int l_kernel = index / output_v / kernel_w / kernel_h % kernel_l;
        const int c_in = index / output_v / kernel_v % input_c;


        const int l_in = l_out * stride_l - pad_l;
        const int h_in = h_out * stride_h - pad_h;
        const int w_in = w_out * stride_w - pad_w;
        const int g_off = c_in / channel_per_deformable_group;
//        const int deform_group = input_c / channel_per_deformable_group;
//        printf("%d %d %d %d %d %d %d %d %d\n",threadIdx.x,b_in,l_out,h_out,w_out,c_in,l_kernel,h_kernel,w_kernel);

        //(CL'H'W')(L"H"W")
        DType *data_col_base_ptr = data_col +
                                   (c_in * kernel_v +
                                    l_kernel * kernel_h * kernel_w +
                                    h_kernel * kernel_w +
                                    w_kernel) * output_v +
                                   l_out * output_h * output_w +
                                   h_out * output_w +
                                   w_out;
        //CLHW
        const DType *data_in_base_ptr = data_im + c_in * input_v;
        //GL"H"W"
        const DType *data_offset_base_ptr = data_offset + g_off*output_v;
        const int offset = l_out*output_h*output_w+h_out*output_w+w_out;

        const DType l_in_after = l_in + l_kernel + data_offset_base_ptr[offset];
        const DType h_in_after = h_in + h_kernel;
        const DType w_in_after = w_in + w_kernel;
//        printf("%d %f %f %f\n",threadIdx.x,l_in_after,h_in_after,w_in_after);

        DType val = 0;
        if (l_in_after > -1 && h_in_after > -1 && w_in_after > -1 && l_in_after < input_l &&
                h_in_after < input_h && w_in_after < input_w) {
            //interpolation
            val = Tri_Linear(data_in_base_ptr, input_l, input_h, input_w,
                             l_in_after, h_in_after, w_in_after);
        }
        *data_col_base_ptr = val;
    }
}


inline int get_cuda_blocks(const int num_kernel) {
    return (num_kernel + THREAD_PRE_BLOCK - 1) / THREAD_PRE_BLOCK;
}

void deformable_im2col(cudaStream_t stream,
                       const double *data_in, const double *data_offset,
                       const int input_c,
                       const int input_l, const int input_h, const int input_w,
                       const int output_l, const int output_h, const int output_w,
                       const int kernel_l, const int kernel_h, const int kernel_w,
                       const int pad_l, const int pad_h, const int pad_w,
                       const int stride_l, const int stride_h, const int stride_w,
                       const int channel_per_deformable_group, double *data_col) {
    int num_kernels = output_l * output_h * output_w * input_c * kernel_l * kernel_h * kernel_w;
    deformable_im2col_gpu_kernel << < get_cuda_blocks(num_kernels), THREAD_PRE_BLOCK, 0, stream >> > (
            num_kernels, data_in, data_offset,
                    input_c, input_l, input_h, input_w,
                    kernel_l, kernel_h, kernel_w,
                    pad_l, pad_h, pad_w,
                    stride_l, stride_h, stride_w,
                    channel_per_deformable_group,
                    output_l, output_h, output_w,
                    data_col);
}

//---------------------------------------------backward to input---------------------------------------------------
template<typename DType>
__global__ void deformable_col2im_input_gpu_kernel(
        const int num_kernels, const DType *data_col, const DType *data_offset,
        const int input_c, const int input_l, const int input_h, const int input_w,
        const int output_l, const int output_h, const int output_w,
        const int kernel_l, const int kernel_h, const int kernel_w,
        const int pad_l, const int pad_h, const int pad_w,
        const int stride_l, const int stride_h, const int stride_w,
        const int channel_per_deformable_group, DType *grad_im) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < num_kernels;
         index += blockDim.x * gridDim.x) {
        const int input_v = input_l * input_w * input_h;
        const int output_v = output_l * output_h * output_w;
        const int kernel_v = kernel_l * kernel_h * kernel_w;

        //L"H"W"CL'H'W'
        const int w_kernel = index % kernel_w;
        const int h_kernel = index / kernel_w % kernel_h;
        const int l_kernel = index / kernel_w / kernel_h % kernel_l;
        const int c_in = index / kernel_v % input_c;
        const int w_out = index / kernel_v / input_c % output_w;
        const int h_out = index / kernel_v / input_c / output_w % output_h;
        const int l_out = index / kernel_v / input_c / output_w / output_h % output_l;

        const int l_in = l_out * stride_l - pad_l;
        const int h_in = h_out * stride_h - pad_h;
        const int w_in = w_out * stride_w - pad_w;
        const int g_off = c_in / channel_per_deformable_group;
//        const int deform_group = input_c / channel_per_deformable_group;

        //CL'H'W'  L"H"W"
        const DType *data_col_base_ptr = data_col +
                                   (c_in * kernel_v +
                                    l_kernel * kernel_h * kernel_w +
                                    h_kernel * kernel_w +
                                    w_kernel) * output_v +
                                   l_out * output_h * output_w +
                                   h_out * output_w +
                                   w_out;
        //GL"H"W"
        const DType *data_offset_base_ptr = data_offset + g_off*output_v;
        const int offset = l_out*output_h*output_w+h_out*output_w+w_out;
        //CLHW
        DType *grad_in_base_ptr = grad_im + c_in * input_v;
//        printf("%d %d %d %d %d %x\n", threadIdx.x,b_in,c_in,input_c, input_v, grad_in_base_ptr);
        const int data_width_1d = input_w;
        const int data_width_2d = input_h * input_w;
        const DType l_in_after = l_in + l_kernel + data_offset_base_ptr[offset];
        const DType h_in_after = h_in + h_kernel;
        const DType w_in_after = w_in + w_kernel;
//        printf("%d %f %f %f\n", threadIdx.x,l_in_after,h_in_after,w_in_after);
        if (l_in_after > -1 && h_in_after > -1 && w_in_after > -1 && l_in_after < input_l &&
                h_in_after < input_h && w_in_after < input_w) {
            //eight point around
            int l_low = int(l_in_after);
            int l_high = (l_in_after >= input_l - 1 || l_in_after <= 0) ? l_low : l_low + 1;

            int a0 = l_low * data_width_2d + h_in_after * data_width_1d + w_in_after;
            int a1 = l_high * data_width_2d + h_in_after * data_width_1d + w_in_after;

            DType l_length = l_in_after - l_low;
            DType h_length = 1 - l_length;


            //grad for input
            atomicAdd(
                    grad_in_base_ptr + a0,
                    h_length *
                    (*data_col_base_ptr));
            atomicAdd(
                    grad_in_base_ptr + a1,
                    l_length *
                    (*data_col_base_ptr));
        }
    }
}


//template<typename DType>
void deformable_col2im_input(cudaStream_t stream,
                             const double *data_col, const double *data_offset,
                             const int input_c,
                             const int input_l, const int input_h, const int input_w,
                             const int output_l, const int output_h, const int output_w,
                             const int kernel_l, const int kernel_h, const int kernel_w,
                             const int pad_l, const int pad_h, const int pad_w,
                             const int stride_l, const int stride_h, const int stride_w,
                             const int channel_per_deformable_group, double *grad_im) {
    const int num_kernels = output_l * output_h * output_w * input_c * kernel_l * kernel_h * kernel_w;
    deformable_col2im_input_gpu_kernel << < get_cuda_blocks(num_kernels), THREAD_PRE_BLOCK, 0, stream >> > (
            num_kernels, data_col, data_offset,
                    input_c, input_l, input_h, input_w,
                    output_l, output_h, output_w,
                    kernel_l, kernel_h, kernel_w,
                    pad_l, pad_h, pad_w,
                    stride_l, stride_h, stride_w,
                    channel_per_deformable_group, grad_im
    );
}


//--------------------------------------------------backward to offset---------------------------------------------


template<typename DType>
__global__ void deformable_col2im_offset_gpu_kernel(
        const int num_kernels, const DType *data_col, const DType *data_im, const DType *data_offset,
        const int input_c,
        const int input_l, const int input_h, const int input_w,
        const int output_l, const int output_h, const int output_w,
        const int kernel_l, const int kernel_h, const int kernel_w,
        const int pad_l, const int pad_h, const int pad_w,
        const int stride_l, const int stride_h, const int stride_w,
        const int channel_per_deformable_group,
        DType *grad_off) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < num_kernels;
         index += blockDim.x * gridDim.x) {
        //GL"H"W"
        const int input_v = input_l * input_w * input_h;
        const int output_v = output_l * output_h * output_w;
        const int kernel_v = kernel_l * kernel_h * kernel_w;

        const int deform_group = input_c / channel_per_deformable_group;

        const int w_out = index % output_w;
        const int h_out = index / output_w % output_h;
        const int l_out = index / output_w / output_h % output_l;
        const int g_off = index / output_v % deform_group;

        const int l_in = l_out * stride_l - pad_l;
        const int h_in = h_out * stride_h - pad_h;
        const int w_in = w_out * stride_w - pad_w;

        //GL"H"W"
        int offset_base = g_off*output_v;
        int offset = l_out * output_h * output_w +
                     h_out * output_w +
                     w_out;
        const DType *data_offset_base_ptr = data_offset + offset_base;
        DType *grad_offset_base_ptr = grad_off + offset_base + offset;
//        printf("%d %f\n",threadIdx.x, *data_offset_base_ptr);
        DType val = 0;
        for (int i = 0; i < channel_per_deformable_group; ++i)
            for(int l_kernel =0; l_kernel<kernel_l; l_kernel++)
                for(int h_kernel =0; h_kernel<kernel_h; h_kernel++)
                    for(int w_kernel =0; w_kernel<kernel_w; w_kernel++){
                        const int c_in = g_off * channel_per_deformable_group + i;
                        //CL'H'W' L"H"W"
                        const DType *data_col_base_ptr = data_col +
                                               (c_in * kernel_v +
                                                l_kernel * kernel_h * kernel_w +
                                                h_kernel * kernel_w +
                                                w_kernel) * output_v +
                                               l_out * output_h * output_w +
                                               h_out * output_w +
                                               w_out;
                        //CLHW
                        const DType *data_in_base_ptr = data_im + c_in * input_v;

                        const int data_width_1d = input_w;
                        const int data_width_2d = input_h * input_w;
                        const DType l_in_after = l_in + l_kernel + data_offset_base_ptr[offset];
                        const DType h_in_after = h_in + h_kernel;
                        const DType w_in_after = w_in + w_kernel;
                        if (l_in_after > -1 && h_in_after > -1 && w_in_after > -1 && l_in_after < input_l &&
                h_in_after < input_h && w_in_after < input_w) {
                            int l_low = int(l_in_after);
                            int l_high = (l_in_after >= input_l - 1 || l_in_after <= 0) ? l_low : l_low + 1;

                            int a0 = l_low * data_width_2d + h_in_after * data_width_1d + w_in_after;
                            int a1 = l_high * data_width_2d + h_in_after * data_width_1d + w_in_after;

                            DType c0 = data_in_base_ptr[a0];
                            DType c1 = data_in_base_ptr[a1];

                            val += *data_col_base_ptr *
                                   (c1 - c0);
                        }
                    }
        *grad_offset_base_ptr = val;
    }
}


//template<typename DType>
void deformable_col2im_offset(cudaStream_t stream,
                              const double *data_col, const double *data_im, const double *data_offset,
                              const int input_c,
                              const int input_l, const int input_h, const int input_w,
                              const int output_l, const int output_h, const int output_w,
                              const int kernel_l, const int kernel_h, const int kernel_w,
                              const int pad_l, const int pad_h, const int pad_w,
                              const int stride_l, const int stride_h, const int stride_w,
                              const int channel_per_deformable_group,
                              double *grad_offset) {
    const int num_kernels = (input_c / channel_per_deformable_group) * output_l * output_h * output_w;
    deformable_col2im_offset_gpu_kernel << < get_cuda_blocks(num_kernels), THREAD_PRE_BLOCK, 0, stream >> > (
            num_kernels, data_col, data_im, data_offset,
                    input_c, input_l, input_h, input_w,
                    output_l, output_h, output_w,
                    kernel_l, kernel_h, kernel_w,
                    pad_l, pad_h, pad_w,
                    stride_l, stride_h, stride_w,
                    channel_per_deformable_group, grad_offset
    );
}
