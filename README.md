# 3D Deformable Convolution Op of Pytorch

Realize the 2D convolution, 2D and 3D deformable convolution in Pytorch 0.3.0. 

Support the group convolution, dilate convolution, group deformable convolution, which split the channels of the input to several splits, each use the independent offset fields.

The official implementation of [2D deformable convolution](https://github.com/msracver/Deformable-ConvNets)  

An implementation of [2D deformable convolution in Pytorch](https://github.com/1zb/deformable-convolution-pytorch)

## Requirements
Pytorch = 0.3.0

## Usage
Change the path in make.sh to your own path (The path of torch). Compile the op by:

```Bash
bash make.sh
```

The .so is in xx_xx_op dir.

Use the test_function.py to test the results of the forward and the backward function.

Use the test_gradient to check the gradient of the backward. NOTE that only the double-precision version can pass the gradient check, i.e. deform2d_double. The double version is slower, so is just used for checking.  

Use the test_model to check the whole model. 