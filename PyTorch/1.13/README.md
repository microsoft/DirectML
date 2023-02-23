# PyTorch with DirectML Samples <!-- omit in toc -->

For detailed instructions on getting started with PyTorch with DirectML, see [GPU accelerated ML training](https://docs.microsoft.com/en-us/windows/ai/directml/gpu-pytorch-windows).

- [Setup](#setup)
- [Samples](#samples)
- [External Links](#external-links)

## Setup

Follow the steps below to get set up with PyTorch on DirectML.

1.	Download and install [Python 3.8 to 3.10](https://www.python.org/).

2. Clone this repo.	

3. Install **torch-directml**

>⚠️ Since torch-directml 0.1.13.1.*, **torch** and **torchvision** will be installed as dependencies

```ps
pip install torch-directml
```

4. Create a DML Device and Test

```
import torch
import torch_directml
dml = torch_directml.device()
```
>⚠️ Note that device creation has changed in torch-directml 0.1.13 from previous versions. The torch-directml backend is currently mapped to “PrivateUse1." The new `torch_directml.device()` API is a convenient wrapper for creating your tenors on the correct device.

## Samples

The following sample models are included in this repo to help you get started. The sample includes both inference and training scripts, and you can either train the models from scratch or use the supplied pre-trained weights.
* [attenion is all you need- the original transformer model](./attention_is_all_you_need/)
* [yolov3- a real-time object detection model](./yolov3/)
* [squeezenet - a small image classification model](./squeezenet)
* [resnet50 - an image classification model](./resnet50)
* [maskrcnn - an object detection model](./objectDetection/maskrcnn/)
* *more coming soon*

## External Links
* [torch-directml PyPI project](https://pypi.org/project/torch-directml/)
* [PyTorch homepage](https://pytorch.org/)
