# CG & 3D Neural Renderer

This repo uses the PyTorch implementation of the paper [Neural 3D Mesh Renderer](http://hiroharu-kato.com/projects_en/neural_renderer.html) by Hiroharu Kato, Yoshitaka Ushiku, and Tatsuya Harada.
It uses a port of the [original Chainer implementation](https://github.com/hiroharu-kato/neural_renderer) released by the authors.
The porting to Python is realized by [daniilidis-group](https://github.com/daniilidis-group/neural_renderer).

The goal of this project is to combine the provided examples by daniilidis for have a generalized render pipile for 3D model based on [CelebMask-HQ dataset](https://github.com/switchablenorms/CelebAMask-HQ/blob/master/README.md).

## Requirements
Python 3.6 and PyTorch 1.2.0.

**Note**: In some newer PyTorch versions you might see some compilation errors involving AT_ASSERT. In these cases you can use the version of the code that is in the branch *at_assert_fix*. These changes will be merged into master in the near future.
## Installation
You can install the package by running
```
pip install neural_renderer_pytorch
```
```
pip install pytorch=1.2.0
```
