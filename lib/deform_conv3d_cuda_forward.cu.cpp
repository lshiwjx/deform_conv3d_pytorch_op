//
// Created by lshi on 17-9-25.
//

#include "deform_conv3d_cuda_forward.cu.h"

//interpolation
template<typename DType>
__device__ DType Tri_Linear(const DType *bottom_data,
                            const int length, const int height, const int width,
                            const double l, const double h, const double w) {
    //length and area
    const int data_width_1d = width;
    const int data_width_2d = height * width;

    //get the cube, the function floor can not be used in template
    int l_low = floor(l);
    int h_low = floor(h);
    int w_low = floor(w);
    int l_high = l_low == l ? l_low : l_low + 1;
    int h_high = h_low == h ? h_low : h_low + 1;
    int w_high = w_low == w ? w_low : w_low + 1;

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
        const int batch_size, const int input_c, const int input_l, const int input_h, const int input_w,
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
        //NL"H"W"CL'H'W'
        const int w_kernel = index % kernel_w;
        const int h_kernel = index / kernel_w % kernel_h;
        const int l_kernel = index / kernel_w / kernel_h % kernel_l;
        const int c_in = index / kernel_v % input_c;
        const int w_out = index / kernel_v / input_c % output_w;
        const int h_out = index / kernel_v / input_c / output_w % output_h;
        const int l_out = index / kernel_v / input_c / output_w / output_h % output_l;
        const int b_in = index / kernel_v / input_c / output_w / output_h / output_l % batch_size;

        const int l_in = l_out * stride_l - pad_l;
        const int h_in = h_out * stride_h - pad_h;
        const int w_in = w_out * stride_w - pad_w;
        const int g_off = c_in / channel_per_deformable_group;
        const int deform_group = input_c / channel_per_deformable_group;

        //CL'H'W'  NL"H"W"
        const DType *data_col_base_ptr = data_col +
                                         (b_in * output_v * input_c +
                                          l_out * output_h * output_w * input_c +
                                          h_out * output_w * input_c +
                                          w_out * input_c +
                                          c_in) +
                                         l_kernel * kernel_h * kernel_w +
                                         h_kernel * kernel_w +
                                         w_kernel;
        //NCLHW
        const DType *data_in_base_ptr = data_im +
                                        b_in * input_c * input_v +
                                        c_in * input_v;
        //NGL'H'W'3L"H"W"
        const DType *data_offset_base_ptr = data_offset +
                                            (b_in * deform_group * kernel_v +
                                             g_off * kernel_v +
                                             l_kernel * kernel_h * kernel_w +
                                             h_kernel * kernel_w +
                                             w_kernel) * 3 * output_v;


        const int offset = l_out * output_h * output_w +
                           h_out * output_w +
                           w_out;
        const DType l_in_after = l_in + l_kernel + data_offset_base_ptr[offset + output_v * 0];
        const DType h_in_after = h_in + h_kernel + data_offset_base_ptr[offset + output_v * 1];
        const DType w_in_after = w_in + w_kernel + data_offset_base_ptr[offset + output_v * 2];

        DType val = 0;
        if (l_in_after >= 0 && h_in_after >= 0 && w_in_after >= 0 && l_in_after <= input_l - 1 &&
            h_in_after <= input_h - 1 && w_in_after <= input_w - 1) {
            //interpolation
            val = Tri_Linear(data_in_base_ptr, input_l, input_h, input_w,
                             l_in_after, h_in_after, w_in_after);
        }
        *data_col_ptr = val;
    }
}

//inline int input_to_output(int input, int pad, int kernel, int stride) {
//    return (input + 2 * pad - kernel) / stride + 1;
//}

inline int get_cuda_blocks(const int num_kernel) {
    return (num_kernel + 1024 - 1) / 1024;
}

template<typename DType>
void deformable_im2col(cudaStream_t stream,
                       const DType *data_in, const DType *data_offset,
                       const int batch_size, const int input_c,
                       const int input_l, const int input_h, const int input_w,
                       const int output_l, const int output_h, const int output_w,
                       const int kernel_l, const int kernel_h, const int kernel_w,
                       const int pad_l, const int pad_h, const int pad_w,
                       const int stride_l, const int stride_h, const int stride_w,
                       const int channel_per_deformable_group, DType *data_col) {
//    int out_l = input_to_output(input_l, pad_l, kernel_l, stride_l);
//    int out_h = input_to_output(input_h, pad_h, kernel_h, stride_h);
//    int out_w = input_to_output(input_w, pad_w, kernel_w, stride_w);
    int num_cuda_kernels = batch_size * input_c * out_l * out_h * out_w * kernel_l * kernel_h * kernel_w;
    deformable_im2col_gpu_kernel << < get_cuda_blocks(num_cuda_kernels), 1024, 0, stream >> > (
            num_kernels, data_in, data_offset,
                    batch_size, input_c, input_l, input_h, input_w,
                    kernel_l, kernel_h, kernel_w,
                    pad_l, pad_h, pad_w,
                    stride_l, stride_h, stride_w,
                    channel_per_deformable_group,
                    output_l, output_h, output_w,
                    data_col);
}

template
void deformable_im2col<float>(cudaStream_t stream,
                              const float *data_in, const float *data_offset,
                              const int batch_size, const int input_c,
                              const int input_l, const int input_h, const int input_w,
                              const int output_l, const int output_h, const int output_w,
                              const int kernel_l, const int kernel_h, const int kernel_w,
                              const int pad_l, const int pad_h, const int pad_w,
                              const int stride_l, const int stride_h, const int stride_w,
                              const int channel_per_deformable_group, float *data_col);

