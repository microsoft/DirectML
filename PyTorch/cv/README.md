# Computer Vision Models for PyTorch with DirectML <!-- omit in toc -->

The following computer vision sample models are included in this repo to help you get started. The sample include both inference and training scripts, and you can either train the models from scratch or use the supplied pre-trained weights.

- [Setup](#setup)
- [Samples](#samples)
- [External Links](#external-links)

## Setup

PyTorch with DirectML is supported on both the latest versions of Windows and the [Windows Subsystem for Linux](https://docs.microsoft.com/windows/wsl/about), and is available for download as a PyPI package. For more information about getting started with `torch-directml`, see our [Windows](https://learn.microsoft.com/windows/ai/directml/pytorch-windows) or [WSL 2](https://learn.microsoft.com/windows/ai/directml/pytorch-wsl) guidance on Microsoft Learn.

Once a Python (3.8 to 3.12) environment is setup, install the latest release of `torch-directml` by running the following command:
```
pip install torch-directml
```

## Samples
* [squeezenet - a small image classification model](./squeezenet)
* [yolov3 - a real-time object detection model](./yolov3/)
* [resnet50 - an image classification model](./resnet50)
* [maskrcnn - an object detection model](./objectDetection/maskrcnn/)
* [torchvision - scripts for training torchvision classification models](./torchvision_classification/)

## External Links

* [torch-directml PyPI project](https://pypi.org/project/torch-directml/)
* [PyTorch homepage](https://pytorch.org/)
