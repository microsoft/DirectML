# PyTorch with DirectML Samples <!-- omit in toc -->

For detailed instructions on getting started with PyTorch with DirectML, see [GPU accelerated ML training](http://aka.ms/gpuinwsldocs).

- [Setup](#setup)
- [Samples](#samples)
- [External Links](#external-links)

## Setup

Follow the steps below to get set up with PyTorch on DirectML.

1.	Download and install [Python 3.8](https://www.python.org/downloads/release/python-380/).

2. Clone this repo.	

3. Install prerequisites
```
    pip install -r pytorch\requirements.txt 
```
4.  Install PyTorch+DirectML
```
    pip install -U --force-reinstall pytorch\pytorch_directml.txt
```

> Note: Currently a warning will be issued when the package is installed that indicates that tensorboard and torchvision reference an incompatible version of torch. This can be safely ignored. The current version of PyTorch on DirecML is based off Torch 1.8.0, and is compatible with these packages.

## Samples

The following sample models are included in this repo to help you get started. The sample includes both inference and training scripts, and you can either train the models from scratch or use the supplied pre-trained weights.

* [squeezenet - a small image classification model](./squeezenet)
* [resnet50 - an image classification model](./resnet50)
* *more coming soon*

## External Links

* [pytorch-directml PyPI project](https://pypi.org/project/pytorch-directml/)
* [PyTorch homepage](https://pytorch.org/)
