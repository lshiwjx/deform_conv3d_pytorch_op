//
// Created by lshi on 17-9-25.
//

#include "deform_conv3d_cuda_backward.cu.h"
#include "deform_conv3d_cuda_forward.cu.h"

template<typename DType>
__global__ void deformable_col2im_input_gpu_kernel(
        const int num_kernels, const DType *data_col, const DType *data_offset,
        const int batch_size, const int input_c,
        const int input_l, const int input_h, const int input_w,
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
                                          c_in) * kernel_v +
                                         l_kernel * kernel_h * kernel_w +
                                         h_kernel * kernel_w +
                                         w_kernel;
        //NGL'H'W'3L"H"W"
        int offset_base = (b_in * deform_group * output_v +
                           g_off * output_v +
                           l_kernel * kernel_h * kernel_w +
                           h_kernel * kernel_w +
                           w_kernel) * output_v * 3;
        int offset = l_out * output_h * output_w +
                     h_out * output_w +
                     w_out;
        const DType *data_offset_base_ptr = data_offset + offset_base;
        //NCLHW
        const DType *grad_in_base_ptr = grad_im +
                                        b_in * input_c * input_v +
                                        c_in * input_v;

        const int data_width_1d = input_w;
        const int data_width_2d = input_h * input_w;
        const DType l_in_after = l_in + l_kernel + data_offset_base_ptr[0 * output_v + offset];
        const DType h_in_after = h_in + h_kernel + data_offset_base_ptr[1 * output_v + offset];
        const DType w_in_after = w_in + w_kernel + data_offset_base_ptr[2 * output_v + offset];
        if (l_in_after >= 0 && h_in_after >= 0 && w_in_after >= 0 && l_in_after <= input_l - 1 &&
            h_in_after <= input_h - 1 && w_in_after <= input_w - 1) {
            //eight point around
            int l_low = floor(l_in_after);
            int h_low = floor(h_in_after);
            int w_low = floor(w_in_after);

            int l_high = l_low == l_in_after ? l_low : l_low + 1;
            int h_high = h_low == h_in_after ? h_low : h_low + 1;
            int w_high = w_low == w_in_after ? w_low : w_low + 1;

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
                             const int batch_size, const int input_c,
                             const int input_l, const int input_h, const int input_w,
                             const int output_l, const int output_h, const int output_w,
                             const int kernel_l, const int kernel_h, const int kernel_w,
                             const int pad_l, const int pad_h, const int pad_w,
                             const int stride_l, const int stride_h, const int stride_w,
                             const int channel_per_deformable_group, DType *grad_im) {
    const int num_kernels = batch_size * input_c * output_l * output_h * output_w * kernel_l * kernel_h * kernel_w;
    deformable_col2im_input_gpu_kernel << < get_cuda_blocks(num_kernels), 1024, 0, stream >> > (
            num_kernels, data_col, data_offset,
                    batch_size, input_c, input_l, input_h, input_w,
                    output_l, output_h, output_w,
                    kernel_l, kernel_h, kernel_w,
                    pad_l, pad_h, pad_w,
                    stride_l, stride_h, stride_w,
                    channel_per_deformable_group, grad_im
    );
}

template
void deformable_col2im_input<float>(cudaStream_t stream,
                                    const float *data_col, const float *data_offset,
                                    const int batch_size, const int input_c,
                                    const int input_l, const int input_h, const int input_w,
                                    const int output_l, const int output_h, const int output_w,
                                    const int kernel_l, const int kernel_h, const int kernel_w,
                                    const int pad_l, const int pad_h, const int pad_w,
                                    const int stride_l, const int stride_h, const int stride_w,
                                    const int channel_per_deformable_group, float *grad_im);

//--------------------------------------------------------------------------------------------------------------


template<typename DType>
__global__ void deformable_col2im_offset_gpu_kernel(
        const int num_kernels, const DType *data_col, const DType *data_im, const DType *data_offset,
        const int batch_size, const int input_c,
        const int input_l, const int input_h, const int input_w,
        const int output_l, const int output_h, const int output_w,
        const int kernel_l, const int kernel_h, const int kernel_w,
        const int pad_l, const int pad_h, const int pad_w,
        const int stride_l, const int stride_h, const int stride_w,
        const int channel_per_deformable_group,
        DType *grad_off) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < num_kernels;
         index += blockDim.x * gridDim.x) {
        //NGL'H'W'3L"H"W"
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
        const int b_in = index / output_v / 3 / kernel_v / deform_group % batch_size;


        const int l_in = l_out * stride_l - pad_l;
        const int h_in = h_out * stride_h - pad_h;
        const int w_in = w_out * stride_w - pad_w;

        //NGL"H"W"L'H'W'3
        int offset_base = (b_in * deform_group * output_v +
                           g_off * output_v +
                           l_kernel * kernel_h * kernel_w +
                           h_kernel * kernel_w +
                           w_kernel) * output_v * 3;
        int offset = l_out * output_h * output_w +
                     h_out * output_w +
                     w_out;
        const DType *data_offset_base_ptr = data_offset + offset_base;
        const DType *grad_offset_base_ptr = grad_off + offset_base + int_3 * output_v + offset;

        DType val = 0;

        for (int i = 0; i < channel_per_deformable_group; ++i) {
            const int c_in = g_off * channel_per_deformable_group + i;
            //CL'H'W'  NL"H"W"
            const DType *data_col_base_ptr = data_col +
                                             (b_in * output_v * input_c +
                                              l_out * output_h * output_w * input_c +
                                              h_out * output_w * input_c +
                                              w_out * input_c +
                                              c_in) * kernel_v +
                                             l_kernel * kernel_h * kernel_w +
                                             h_kernel * kernel_w +
                                             w_kernel;
            //NCLHW
            const DType *data_in_base_ptr = data_im +
                                            b_in * input_c * input_v +
                                            c_in * input_v;


            const int data_width_1d = input_w;
            const int data_width_2d = input_h * input_w;
            const DType l_in_after = l_in + l_kernel + data_offset_base_ptr[0 * output_v + offset];
            const DType h_in_after = h_in + h_kernel + data_offset_base_ptr[1 * output_v + offset];
            const DType w_in_after = w_in + w_kernel + data_offset_base_ptr[2 * output_v + offset];
            if (l_in_after >= 0 && h_in_after >= 0 && w_in_after >= 0 && l_in_after <= input_l - 1 &&
                h_in_after <= input_h - 1 && w_in_after <= input_w - 1) {
                //eight point around
                int l_low = floor(l_in_after);
                int h_low = floor(h_in_after);
                int w_low = floor(w_in_after);

                int l_high = l_low == l_in_after ? l_low : l_low + 1;
                int h_high = h_low == h_in_after ? h_low : h_low + 1;
                int w_high = w_low == w_in_after ? w_low : w_low + 1;

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
        }
        *grad_offset_base_ptr = val;
    }
}


template<typename DType>
void deformable_col2im_offset(cudaStream_t stream,
                              const DType *data_col, const DType *data_im, const DType *data_offset,
                              const int batch_size, const int input_c,
                              const int input_l, const int input_h, const int input_w,
                              const int output_l, const int output_h, const int output_w,
                              const int kernel_l, const int kernel_h, const int kernel_w,
                              const int pad_l, const int pad_h, const int pad_w,
                              const int stride_l, const int stride_h, const int stride_w,
                              const int channel_per_deformable_group,
                              DType *grad_offset) {
    const int num_kernels = batch_size * (input_c / channel_per_deformable_group)
                            * kernel_l * kernel_h * kernel_w * 3 * output_l * output_h * output_w;
    deformable_col2im_offset_gpu_kernel << < get_cuda_blocks(num_kernels), 1024, 0, stream >> > (
            num_kernels, data_col, data_im, data_offset,
                    batch_size, input_c, input_l, input_h, input_w,
                    output_l, output_h, output_w,
                    kernel_l, kernel_h, kernel_w,
                    pad_l, pad_h, pad_w,
                    stride_l, stride_h, stride_w,
                    channel_per_deformable_group, grad_offset
    );
}


template
void deformable_col2im_offset<float>(cudaStream_t stream,
                                     const float *data_col, const float *data_im, const float *data_offset,
                                     const int batch_size, const int input_c,
                                     const int input_l, const int input_h, const int input_w,
                                     const int output_l, const int output_h, const int output_w,
                                     const int kernel_l, const int kernel_h, const int kernel_w,
                                     const int pad_l, const int pad_h, const int pad_w,
                                     const int stride_l, const int stride_h, const int stride_w,
                                     const int channel_per_deformable_group,
                                     float *grad_offset);