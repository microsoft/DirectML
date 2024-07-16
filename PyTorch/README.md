# PyTorch with DirectML Samples <!-- omit in toc -->

PyTorch with DirectML enables training and inference of complex machine learning models on a wide range of DirectX 12-compatible hardware. This is done through [`torch-directml`](https://pypi.org/project/torch-directml/), a plugin for PyTorch.

DirectML is a high-performance, hardware-accelerated DirectX 12 library for machine learning. DirectML provides GPU acceleration for common machine learning tasks across a broad range of supported hardware and drivers, including all DirectX 12-capable GPUs from vendors such as AMD, Intel, NVIDIA, and Qualcomm.

More information about DirectML can be found on the [DirectML Overview](https://learn.microsoft.com/windows/ai/directml/dml) page on Microsoft Learn.

## Setup
PyTorch with DirectML is supported on both the latest versions of Windows and the [Windows Subsystem for Linux](https://docs.microsoft.com/windows/wsl/about), and is available for download as a PyPI package. For more information about getting started with `torch-directml`, see our [Windows](https://learn.microsoft.com/windows/ai/directml/pytorch-windows) or [WSL 2](https://learn.microsoft.com/windows/ai/directml/pytorch-wsl) guidance on Microsoft Learn.

Once a Python (3.8 to 3.12) environment is setup, install the latest release of `torch-directml` by running the following command:
```
pip install torch-directml
```

## Samples
Try the `torch-directml` samples below or explore the [cv](./cv/), [transformer](./transformer/), [llm](./llm/) and [diffusion](./diffusion/) folders:
* [attention is all you need - the original transformer model](./transformer/attention_is_all_you_need/)
* [yolov3 - a real-time object detection model](./cv/yolov3/)
* [squeezenet - a small image classification model](./cv/squeezenet)
* [resnet50 - an image classification model](./cv/resnet50)
* [maskrcnn - an object detection model](./cv/objectDetection/maskrcnn/)
* [llm - a text generation and chatbot app supporting various language models](./llm/)
* [whisper - a general-purpose speech recognition model](./audio/whisper/)
* [Stable Diffusion Turbo & XL Turbo - a text-to-image generation model](./diffusion/sd/)

## External Links
* [torch-directml PyPI project](https://pypi.org/project/torch-directml/)
* [PyTorch homepage](https://pytorch.org/)
