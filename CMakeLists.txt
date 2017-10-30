cmake_minimum_required(VERSION 3.7)
project(deform3d_new_pytorch)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 11)

include_directories(
        "/home/lshi/Application/Anaconda/envs/pytorch/lib/python3.6/site-packages/torch/lib/include"
        "/usr/local/cuda/include"
        "./"
)

add_subdirectory(deform3d/lib)
add_subdirectory(deform2d/lib)