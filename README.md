# 3D Deformable Convolution Op of Pytorch

Realize the 2D convolution, 2D and 3D deformable convolution. Support the group, dilation, group deform.
The official implementation of [2D deformable convolution](https://github.com/msracver/Deformable-ConvNets)  
An implementation of [2D deformable convolution in Pytorch](https://github.com/1zb/deformable-convolution-pytorch)

## Requirements
Pytorch = 0.3

## Usage
Change the path in make.sh to your own path. Compile the op by:

```Bash
bash make.sh
```
The double-precision version can pass the gradient check, i.e. test_gradient.py. 