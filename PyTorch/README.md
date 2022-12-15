# PyTorch with DirectML Samples <!-- omit in toc -->

DirectML acceleration for PyTorch is currently available for Public Preview. PyTorch with DirectML enables training and inference of complex machine learning models on a wide range of DirectX 12-compatible hardware.

DirectML is a high-performance, hardware-accelerated DirectX 12 library for machine learning. DirectML provides GPU acceleration for common machine learning tasks across a broad range of supported hardware and drivers, including all DirectX 12-capable GPUs from vendors such as AMD, Intel, NVIDIA, and Qualcomm.

More information about DirectML can be found in [Introduction to DirectML](https://docs.microsoft.com/windows/win32/direct3d12/dml-intro).

PyTorch on DirectML is supported on both the latest versions of Windows 10 and the [Windows Subsystem for Linux](https://docs.microsoft.com/windows/wsl/about), and is available for download as a PyPI package. For more information about getting started, see [GPU accelerated ML training (docs.microsoft.com)](http://aka.ms/gpuinwsldocs)

## Pytorch with DirectML Versions
| torch-directml        | pytorch |
|-----------------------|-------|
| [0.1.13.\*](https://pypi.org/project/torch-directml/)                | 1.13  |
| [1.8.0a0.\*](https://pypi.org/project/pytorch-directml/) | 1.8   |

## Setup
* For users of Pytorch-DirectML forked from Pytorch __1.13__, see the setup instructions in the [1.13](./1.13/) folder. 
* For users of Pytorch-DirectML forked from Pytorch __1.8__, see the setup instructions in the [1.8](./1.8/) folder.

## Samples
For users of Pytorch-DirectML forked from Pytorch 1.13, the samples can be found below or in the [1.13](./1.13/) folder: 
* [attenion is all you need- the original transformer model](./1.13/attention_is_all_you_need/)
* [yolov3- a real-time object detection model](./1.13/yolov3/)
* [squeezenet - a small image classification model](./1.13/squeezenet)
* [resnet50 - an image classification model](./1.13/resnet50)
* [maskrcnn - an object detection model](./1.13/objectDetection/maskrcnn/)

For users of Pytorch-DirectML forked from Pytorch 1.8, the samples can be found below or in the [1.8](./1.8/) folder: 
* [squeezenet - a small image classification model](./1.8/squeezenet)
* [resnet50 - an image classification model](./1.8/resnet50)
* [maskrcnn - an object detection model](./1.8/objectDetection/maskrcnn/)

## External Links

* [PyTorch homepage](https://pytorch.org/)
* [torch-directml PyPI project](https://pypi.org/project/torch-directml/)
