# PyTorch with DirectML Samples <!-- omit in toc -->

For detailed instructions on getting started with PyTorch with DirectML, see [GPU accelerated ML training](https://docs.microsoft.com/en-us/windows/ai/directml/gpu-pytorch-windows).

- [Setup](#setup)
- [Samples](#samples)
- [External Links](#external-links)

## Setup

Follow the steps below to get set up with PyTorch on DirectML.

1.	Download and install [Python 3.8](https://www.python.org/downloads/release/python-380/).

2. Clone this repo.	

3. Install prerequisites
```
    pip install torchvision==0.9.0
    pip uninstall torch
    pip install pytorch-directml
```

> Note: The torchvision package automatically installs the torch==1.8.0 dependency, but this is not needed and will cause collisions with the pytorch-directml package. We must uninstall the torch package after installing requirements.

4. _(optional)_ Run `pip list`. The following packages should be installed:
```
pytorch-directml        1.8.0a0.dev211019
torchvision             0.9.0
```

## Samples

The following sample models are included in this repo to help you get started. The sample includes both inference and training scripts, and you can either train the models from scratch or use the supplied pre-trained weights.

* [squeezenet - a small image classification model](./squeezenet)
* [resnet50 - an image classification model](./resnet50)
* [maskrcnn - an object detection model](./objectDetection/maskrcnn/)
* *more coming soon*

## External Links

* [pytorch-directml PyPI project](https://pypi.org/project/pytorch-directml/)
* [PyTorch homepage](https://pytorch.org/)
