import os.path as path
import torch
from torch.utils.ffi import create_extension

sources = ['lib/deform_conv3d_cuda.cpp']
headers = ['lib/deform_conv3d_cuda.h']
defines = [('WITH_CUDA', None)]
with_cuda = True

this_file = path.dirname(path.realpath(__file__))
print(this_file)
extra_objects = ['lib/deform_conv3d_cuda_kernel.cu.o']
extra_objects = [path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
    'deform_conv3d_op',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects
)

if __name__ == '__main__':
    assert torch.cuda.is_available(), 'Please install CUDA for GPU support.'
    ffi.build()
