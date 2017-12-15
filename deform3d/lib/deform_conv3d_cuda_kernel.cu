//
// Created by lshi on 17-9-25.
//

#include "deform_conv3d_cuda_kernel.h"
#include "THC/THC.h"
#include "TH/TH.h"

#define THREAD_PRE_BLOCK 1024

//--------------------------------------------------forward---------------------------------------------
template<typename DType>
__device__ DType Tri_Linear(const DType *bottom_data,
                            const int length, const int height, const int width,
                            const double l, const double h, const double w) {
    //length and area
    const int data_width_1d = width;
    const int data_width_2d = height * width;

    //get the cube, the function int can not be used in template
    int l_low = int(l);
    int h_low = int(h);
    int w_low = int(w);
    int l_high = (l >= length - 1 || l <= 0) ? l_low : l_low + 1;
    int h_high = (h >= height - 1 || h <= 0) ? h_low : h_low + 1;
    int w_high = (w >= width - 1 || w <= 0) ? w_low : w_low + 1;

    //the corner, format is lhw
    DType c000 = bottom_data[l_low * data_width_2d + h_low * data_width_1d + w_low];
    DType c001 = bottom_data[l_low * data_width_2d + h_low * data_width_1d + w_high];
    DType c010 = bottom_data[l_low * data_width_2d + h_high * data_width_1d + w_low];
    DType c011 = bottom_data[l_low * data_width_2d + h_high * data_width_1d + w_high];

    DType c100 = bottom_data[l_high * data_width_2d + h_low * data_width_1d + w_low];
    DType c101 = bottom_data[l_high * data_width_2d + h_low * data_width_1d + w_high];
    DType c110 = bottom_data[l_high * data_width_2d + h_high * data_width_1d + w_low];
    DType c111 = bottom_data[l_high * data_width_2d + h_high * data_width_1d + w_high];

    //calculate the distance between the point and corner, using 1 to make sure using the low if equal
    DType l_width = w - w_low;
    DType h_width = 1 - l_width;
    DType l_height = h - h_low;
    DType h_height = 1 - l_height;
    DType l_length = l - l_low;
    DType h_length = 1 - l_length;

    //interpolation
    DType c00 = c000 * h_width + c001 * l_width;
    DType c01 = c010 * h_width + c011 * l_width;
    DType c10 = c100 * h_width + c101 * l_width;
    DType c11 = c110 * h_width + c111 * l_width;

    DType c0 = c00 * h_height + c01 * l_height;
    DType c1 = c10 * h_height + c11 * l_height;

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
        const int dilation_l, const int dilation_h, const int dilation_w,
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
        //GL'H'W'3L"H"W"
        const DType *data_offset_base_ptr = data_offset +
                                            (g_off * kernel_v +
                                             l_kernel * kernel_h * kernel_w +
                                             h_kernel * kernel_w +
                                             w_kernel) * 3 * output_v;

        const int offset = l_out * output_h * output_w +
                           h_out * output_w +
                           w_out;
//        printf("%d %d %d %d %f %f %f\n",threadIdx.x,l_in,h_in,w_in,data_offset_base_ptr[offset + output_v * 0],
//               data_offset_base_ptr[offset + output_v * 1],data_offset_base_ptr[offset + output_v * 2]);
        const DType l_in_after = l_in + l_kernel*dilation_l + data_offset_base_ptr[offset + output_v * 0];
        const DType h_in_after = h_in + h_kernel*dilation_h + data_offset_base_ptr[offset + output_v * 1];
        const DType w_in_after = w_in + w_kernel*dilation_w + data_offset_base_ptr[offset + output_v * 2];
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

template<typename DType>
void deformable_im2col(cudaStream_t stream,
                       const DType *data_in, const DType *data_offset,
                       const int input_c,
                       const int input_l, const int input_h, const int input_w,
                       const int output_l, const int output_h, const int output_w,
                       const int kernel_l, const int kernel_h, const int kernel_w,
                       const int pad_l, const int pad_h, const int pad_w,
                       const int stride_l, const int stride_h, const int stride_w,
                       const int dilation_l, const int dilation_h, const int dilation_w,
                       const int channel_per_deformable_group, DType *data_col) {
    int num_kernels = output_l * output_h * output_w * input_c * kernel_l * kernel_h * kernel_w;
    deformable_im2col_gpu_kernel << < get_cuda_blocks(num_kernels), THREAD_PRE_BLOCK, 0, stream >> > (
            num_kernels, data_in, data_offset,
                    input_c, input_l, input_h, input_w,
                    kernel_l, kernel_h, kernel_w,
                    pad_l, pad_h, pad_w,
                    stride_l, stride_h, stride_w,
                    dilation_l, dilation_h, dilation_w,
                    channel_per_deformable_group,
                    output_l, output_h, output_w,
                    data_col);
}

template
void deformable_im2col<float>(cudaStream_t stream,
                              const float *data_in, const float *data_offset,
                              const int input_c,
                              const int input_l, const int input_h, const int input_w,
                              const int output_l, const int output_h, const int output_w,
                              const int kernel_l, const int kernel_h, const int kernel_w,
                              const int pad_l, const int pad_h, const int pad_w,
                              const int stride_l, const int stride_h, const int stride_w,
                              const int dilation_l, const int dilation_h, const int dilation_w,
                              const int channel_per_deformable_group, float *data_col);

//---------------------------------------------backward to input---------------------------------------------------
template<typename DType>
__global__ void deformable_col2im_input_gpu_kernel(
        const int num_kernels, const DType *data_col, const DType *data_offset,
        const int input_c, const int input_l, const int input_h, const int input_w,
        const int output_l, const int output_h, const int output_w,
        const int kernel_l, const int kernel_h, const int kernel_w,
        const int pad_l, const int pad_h, const int pad_w,
        const int stride_l, const int stride_h, const int stride_w,
        const int dilation_l, const int dilation_h, const int dilation_w,
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
        //GL'H'W'3L"H"W"
        int offset_base = (g_off * kernel_v +
                           l_kernel * kernel_h * kernel_w +
                           h_kernel * kernel_w +
                           w_kernel) * output_v * 3;
        int offset = l_out * output_h * output_w +
                     h_out * output_w +
                     w_out;
        const DType *data_offset_base_ptr = data_offset + offset_base;
        //CLHW
        DType *grad_in_base_ptr = grad_im + c_in * input_v;
//        printf("%d %d %d %d %d %x\n", threadIdx.x,b_in,c_in,input_c, input_v, grad_in_base_ptr);
        const int data_width_1d = input_w;
        const int data_width_2d = input_h * input_w;
        const DType l_in_after = l_in + l_kernel*dilation_l + data_offset_base_ptr[0 * output_v + offset];
        const DType h_in_after = h_in + h_kernel*dilation_h + data_offset_base_ptr[1 * output_v + offset];
        const DType w_in_after = w_in + w_kernel*dilation_w + data_offset_base_ptr[2 * output_v + offset];
//        printf("%d %f %f %f\n", threadIdx.x,l_in_after,h_in_after,w_in_after);
        if (l_in_after > -1 && h_in_after > -1 && w_in_after > -1 && l_in_after < input_l &&
            h_in_after < input_h && w_in_after < input_w) {
            //eight point around
            int l_low = int(l_in_after);
            int h_low = int(h_in_after);
            int w_low = int(w_in_after);

            int l_high = (l_in_after >= input_l - 1 || l_in_after <= 0) ? l_low : l_low + 1;
            int h_high = (h_in_after >= input_h - 1 || h_in_after <= 0) ? h_low : h_low + 1;
            int w_high = (w_in_after >= input_w - 1 || w_in_after <= 0) ? w_low : w_low + 1;

            int a000 = l_low * data_width_2d + h_low * data_width_1d + w_low;
            int a001 = l_low * data_width_2d + h_low * data_width_1d + w_high;
            int a010 = l_low * data_width_2d + h_high * data_width_1d + w_low;
            int a011 = l_low * data_width_2d + h_high * data_width_1d + w_high;
            int a100 = l_high * data_width_2d + h_low * data_width_1d + w_low;
            int a101 = l_high * data_width_2d + h_low * data_width_1d + w_high;
            int a110 = l_high * data_width_2d + h_high * data_width_1d + w_low;
            int a111 = l_high * data_width_2d + h_high * data_width_1d + w_high;

            //six distance
            DType l_width = w_in_after - w_low;
            DType h_width = 1 - l_width;
            DType l_height = h_in_after - h_low;
            DType h_height = 1 - l_height;
            DType l_length = l_in_after - l_low;
            DType h_length = 1 - l_length;



            //grad for input
            atomicAdd(
                    grad_in_base_ptr + a000,
                    h_length * h_height * h_width *
                    (*data_col_base_ptr));
            atomicAdd(
                    grad_in_base_ptr + a001,
                    h_length * h_height * l_width *
                    (*data_col_base_ptr));
            atomicAdd(
                    grad_in_base_ptr + a010,
                    h_length * l_height * h_width *
                    (*data_col_base_ptr));
            atomicAdd(
                    grad_in_base_ptr + a011,
                    h_length * l_height * l_width *
                    (*data_col_base_ptr));
            atomicAdd(
                    grad_in_base_ptr + a100,
                    l_length * h_height * h_width *
                    (*data_col_base_ptr));
            atomicAdd(
                    grad_in_base_ptr + a101,
                    l_length * h_height * l_width *
                    (*data_col_base_ptr));
            atomicAdd(
                    grad_in_base_ptr + a110,
                    l_length * l_height * h_width *
                    (*data_col_base_ptr));
            atomicAdd(
                    grad_in_base_ptr + a111,
                    l_length * l_height * l_width *
                    (*data_col_base_ptr));
        }
    }
}


template<typename DType>
void deformable_col2im_input(cudaStream_t stream,
                             const DType *data_col, const DType *data_offset,
                             const int input_c,
                             const int input_l, const int input_h, const int input_w,
                             const int output_l, const int output_h, const int output_w,
                             const int kernel_l, const int kernel_h, const int kernel_w,
                             const int pad_l, const int pad_h, const int pad_w,
                             const int stride_l, const int stride_h, const int stride_w,
                             const int dilation_l, const int dilation_h, const int dilation_w,
                             const int channel_per_deformable_group, DType *grad_im) {
    const int num_kernels = output_l * output_h * output_w * input_c * kernel_l * kernel_h * kernel_w;
    deformable_col2im_input_gpu_kernel << < get_cuda_blocks(num_kernels), THREAD_PRE_BLOCK, 0, stream >> > (
            num_kernels, data_col, data_offset,
                    input_c, input_l, input_h, input_w,
                    output_l, output_h, output_w,
                    kernel_l, kernel_h, kernel_w,
                    pad_l, pad_h, pad_w,
                    stride_l, stride_h, stride_w,
                    dilation_l, dilation_h, dilation_w,
                    channel_per_deformable_group, grad_im
    );
}

template
void deformable_col2im_input<float>(cudaStream_t stream,
                                    const float *data_col, const float *data_offset,
                                    const int input_c,
                                    const int input_l, const int input_h, const int input_w,
                                    const int output_l, const int output_h, const int output_w,
                                    const int kernel_l, const int kernel_h, const int kernel_w,
                                    const int pad_l, const int pad_h, const int pad_w,
                                    const int stride_l, const int stride_h, const int stride_w,
                                    const int dilation_l, const int dilation_h, const int dilation_w,
                                    const int channel_per_deformable_group, float *grad_im);

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
        const int dilation_l, const int dilation_h, const int dilation_w,
        const int channel_per_deformable_group,
        DType *grad_off) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < num_kernels;
         index += blockDim.x * gridDim.x) {
        //GL'H'W'3L"H"W"
        const int input_v = input_l * input_w * input_h;
        const int output_v = output_l * output_h * output_w;
        const int kernel_v = kernel_l * kernel_h * kernel_w;

        const int deform_group = input_c / channel_per_deformable_group;

        const int w_out = index % output_w;
        const int h_out = index / output_w % output_h;
        const int l_out = index / output_w / output_h % output_l;
        const int int_3 = index / output_v % 3;
        const int w_kernel = index / output_v / 3 % kernel_w;
        const int h_kernel = index / output_v / 3 / kernel_w % kernel_h;
        const int l_kernel = index / output_v / 3 / kernel_w / kernel_h % kernel_l;
        const int g_off = index / output_v / 3 / kernel_v % deform_group;

        const int l_in = l_out * stride_l - pad_l;
        const int h_in = h_out * stride_h - pad_h;
        const int w_in = w_out * stride_w - pad_w;

        //GL"H"W"L'H'W'3
        int offset_base = (g_off * kernel_v +
                           l_kernel * kernel_h * kernel_w +
                           h_kernel * kernel_w +
                           w_kernel) * output_v * 3;
        int offset = l_out * output_h * output_w +
                     h_out * output_w +
                     w_out;
        const DType *data_offset_base_ptr = data_offset + offset_base;
        DType *grad_offset_base_ptr = grad_off + offset_base + int_3 * output_v + offset;
//        printf("%d %f\n",threadIdx.x, *data_offset_base_ptr);
        DType val = 0;
        for (int i = 0; i < channel_per_deformable_group; ++i) {
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
            const DType l_in_after = l_in + l_kernel*dilation_l + data_offset_base_ptr[0 * output_v + offset];
            const DType h_in_after = h_in + h_kernel*dilation_h + data_offset_base_ptr[1 * output_v + offset];
            const DType w_in_after = w_in + w_kernel*dilation_w + data_offset_base_ptr[2 * output_v + offset];
            if (l_in_after > -1 && h_in_after > -1 && w_in_after > -1 && l_in_after < input_l &&
                h_in_after < input_h && w_in_after < input_w) {
                //eight point around
                int l_low = int(l_in_after);
                int h_low = int(h_in_after);
                int w_low = int(w_in_after);

                int l_high = (l_in_after >= input_l - 1 || l_in_after <= 0) ? l_low : l_low + 1;
                int h_high = (h_in_after >= input_h - 1 || h_in_after <= 0) ? h_low : h_low + 1;
                int w_high = (w_in_after >= input_w - 1 || w_in_after <= 0) ? w_low : w_low + 1;

                int a000 = l_low * data_width_2d + h_low * data_width_1d + w_low;
                int a001 = l_low * data_width_2d + h_low * data_width_1d + w_high;
                int a010 = l_low * data_width_2d + h_high * data_width_1d + w_low;
                int a011 = l_low * data_width_2d + h_high * data_width_1d + w_high;
                int a100 = l_high * data_width_2d + h_low * data_width_1d + w_low;
                int a101 = l_high * data_width_2d + h_low * data_width_1d + w_high;
                int a110 = l_high * data_width_2d + h_high * data_width_1d + w_low;
                int a111 = l_high * data_width_2d + h_high * data_width_1d + w_high;

                //value of eight point
                DType c000 = data_in_base_ptr[a000];
                DType c001 = data_in_base_ptr[a001];
                DType c010 = data_in_base_ptr[a010];
                DType c011 = data_in_base_ptr[a011];

                DType c100 = data_in_base_ptr[a100];
                DType c101 = data_in_base_ptr[a101];
                DType c110 = data_in_base_ptr[a110];
                DType c111 = data_in_base_ptr[a111];

                //six distance
                DType l_width = w_in_after - w_low;
                DType h_width = 1 - l_width;
                DType l_height = h_in_after - h_low;
                DType h_height = 1 - l_height;
                DType l_length = l_in_after - l_low;
                DType h_length = 1 - l_length;

                switch (int_3) {
                    case 0:
                        val += *data_col_base_ptr *
                               (c100 * h_height * h_width + c101 * h_height * l_width +
                                c110 * l_height * h_width + c111 * l_height * l_width -
                                c000 * h_height * h_width - c001 * h_height * l_width -
                                c010 * l_height * h_width - c011 * l_height * l_width);
                        break;
                    case 1:
                        val += *data_col_base_ptr *
                               (c010 * h_length * h_width + c011 * h_length * l_width +
                                c110 * l_length * h_width + c111 * l_length * l_width -
                                c000 * h_length * h_width - c001 * h_length * l_width -
                                c100 * l_length * h_width - c101 * l_length * l_width);
                        break;
                    case 2:
                        val += *data_col_base_ptr *
                               (c001 * h_height * h_length + c101 * h_height * l_length +
                                c011 * l_height * h_length + c111 * l_height * l_length -
                                c000 * h_height * h_length - c100 * h_height * l_length -
                                c010 * l_height * h_length - c110 * l_height * l_length);
                        break;
                    default:
                        printf("error in switch");
                }
            }
            else if (l_in_after <= -1 && int_3 == 0)
                 val += (*data_col_base_ptr>0?-*data_col_base_ptr:*data_col_base_ptr);
            else if (l_in_after >= input_l && int_3 == 0)
                 val += (*data_col_base_ptr>0?*data_col_base_ptr:-*data_col_base_ptr);
            else if (h_in_after <= -1 && int_3 == 1)
                 val += (*data_col_base_ptr>0?-*data_col_base_ptr:*data_col_base_ptr);
            else if (h_in_after >= input_l && int_3 == 1)
                 val += (*data_col_base_ptr>0?*data_col_base_ptr:-*data_col_base_ptr);
            else if (w_in_after <= -1 && int_3 == 2)
                 val += (*data_col_base_ptr>0?-*data_col_base_ptr:*data_col_base_ptr);
            else if (w_in_after >= input_l && int_3 == 2)
                 val += (*data_col_base_ptr>0?*data_col_base_ptr:-*data_col_base_ptr);
        }
        *grad_offset_base_ptr = val;
    }
}


template<typename DType>
void deformable_col2im_offset(cudaStream_t stream,
                              const DType *data_col, const DType *data_im, const DType *data_offset,
                              const int input_c,
                              const int input_l, const int input_h, const int input_w,
                              const int output_l, const int output_h, const int output_w,
                              const int kernel_l, const int kernel_h, const int kernel_w,
                              const int pad_l, const int pad_h, const int pad_w,
                              const int stride_l, const int stride_h, const int stride_w,
                              const int dilation_l, const int dilation_h, const int dilation_w,
                              const int channel_per_deformable_group,
                              DType *grad_offset) {
    const int num_kernels = (input_c / channel_per_deformable_group)
                            * kernel_l * kernel_h * kernel_w * 3 * output_l * output_h * output_w;
    deformable_col2im_offset_gpu_kernel << < get_cuda_blocks(num_kernels), THREAD_PRE_BLOCK, 0, stream >> > (
            num_kernels, data_col, data_im, data_offset,
                    input_c, input_l, input_h, input_w,
                    output_l, output_h, output_w,
                    kernel_l, kernel_h, kernel_w,
                    pad_l, pad_h, pad_w,
                    stride_l, stride_h, stride_w,
                    dilation_l, dilation_h, dilation_w,
                    channel_per_deformable_group, grad_offset
    );
}


template
void deformable_col2im_offset<float>(cudaStream_t stream,
                                     const float *data_col, const float *data_im, const float *data_offset,
                                     const int input_c,
                                     const int input_l, const int input_h, const int input_w,
                                     const int output_l, const int output_h, const int output_w,
                                     const int kernel_l, const int kernel_h, const int kernel_w,
                                     const int pad_l, const int pad_h, const int pad_w,
                                     const int stride_l, const int stride_h, const int stride_w,
                                     const int dilation_l, const int dilation_h, const int dilation_w,
                                     const int channel_per_deformable_group,
                                     float *grad_offset);
