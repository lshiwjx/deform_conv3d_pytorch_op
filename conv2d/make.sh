#!/usr/bin/env bash
cd lib
nvcc -c -o conv2d_cuda_kernel.cu.o conv2d_cuda_kernel.cu \
     -I/home/lshi/Application/Anaconda/envs/pytorch36/lib/python3.6/site-packages/torch/lib/include/  \
     -I/home/lshi/Application/Anaconda/envs/pytorch36/lib/python3.6/site-packages/torch/lib/include/TH  \
     -x cu -Xcompiler -fPIC -std=c++11 -Wno-deprecated-gpu-targets
cd ..
CC=g++ python build.py