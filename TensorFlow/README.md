# TensorFlow on DirectML <!-- omit in toc -->

DirectML support for TensorFlow 1.15 is experimental and currently available as a Public Preview.

TensorFlow on DirectML enables training and inference of complex machine learning models on a wide range of DirectX 12-compatible hardware.

- [Getting Started](#getting-started)
- [Samples](#samples)
- [Feedback](#feedback)
- [External Links](#external-links)

## Getting Started

TensorFlow on DirectML is supported on both the latest versions of Windows 10 and the [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/about).

For detailed instructions on getting started, see [GPU accelerated ML training (docs.microsoft.com)](http://aka.ms/gpuinwsldocs).

## Samples

Two sample models are included in this repo to help you get started. These samples include both inference and training scripts, and you can either train the models from scratch or use the supplied pre-trained weights.

* [squeezenet - a small image classification model](./squeezenet)
* [yolov3 - real-time object detection model](./yolov3)

## Feedback

For comments, questions, feedback, or if you're having problems, please [file an issue](https://github.com/microsoft/DirectML/issues). Alternatively you can contact us directly at askdirectml@microsoft.com

## External Links

* [tensorflow-directml PyPI project](https://pypi.org/project/tensorflow-directml/)
* [TensorFlow GitHub | RFC: TensorFlow on DirectML](https://github.com/tensorflow/community/pull/243)
* [TensorFlow homepage](https://www.tensorflow.org/)
