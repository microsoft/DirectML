# DirectML <!-- omit in toc -->

DirectML is a high-performance, hardware-accelerated DirectX 12 library for machine learning. DirectML provides GPU acceleration for common machine learning tasks across a broad range of supported hardware and drivers, including all DirectX 12-capable GPUs from vendors such as AMD, Intel, NVIDIA, and Qualcomm.

When used standalone, the DirectML API is a low-level DirectX 12 library and is suitable for high-performance, low-latency applications such as frameworks, games, and other real-time applications. The seamless interoperability of DirectML with Direct3D 12 as well as its low overhead and conformance across hardware makes DirectML ideal for accelerating machine learning when both high performance is desired, and the reliability and predictability of results across hardware is critical.

More information about DirectML can be found in [Introduction to DirectML](https://docs.microsoft.com/en-us/windows/win32/direct3d12/dml-intro).

- [Getting Started with DirectML](#getting-started-with-directml)
  - [Hardware requirements](#hardware-requirements)
  - [For application developers](#for-application-developers)
  - [For users, data scientists, and researchers](#for-users-data-scientists-and-researchers)
- [DirectML Samples](#directml-samples)
- [Windows ML on DirectML](#windows-ml-on-directml)
- [ONNX Runtime on DirectML](#onnx-runtime-on-directml)
- [TensorFlow with DirectML (Preview)](#tensorflow-with-directml-preview)
- [Feedback](#feedback)
- [External Links](#external-links)
  - [Documentation](#documentation)
  - [More information](#more-information)
- [Contributing](#contributing)

## Getting Started with DirectML

DirectML is distributed as a system component of Windows 10, and is available as part of the Windows 10 OS in all versions 1903 (Build 10.0.18362; 19H1, "May 2019 Update") and newer.

### Hardware requirements

DirectML requires a DirectX 12 capable device. Almost all commercially-available graphics cards released in the last several years support DirectX 12. Examples of compatible hardware include:

* AMD GCN 1st Gen (Radeon HD 7000 series) and above
* Intel Haswell (4th-gen core) HD Integrated Graphics and above
* NVIDIA Kepler (GTX 600 series) and above
* Qualcomm Adreno 600 and above

### For application developers

DirectML exposes a native C++ DirectX 12 API. The header and library (DirectML.h/DirectML.lib) are available as part of the Windows 10 SDK version 10.0.18362 or newer.

* The Windows 10 SDK can be downloaded from the [Windows Dev Center](https://developer.microsoft.com/en-us/windows/downloads/windows-10-sdk/)
* [DirectML programming guide](https://docs.microsoft.com/en-us/windows/win32/direct3d12/dml)
* [DirectML API reference](https://docs.microsoft.com/en-us/windows/win32/direct3d12/direct3d-directml-reference)

### For users, data scientists, and researchers

DirectML is built-in as a backend to several frameworks such as Windows ML, ONNX Runtime, and TensorFlow.

See the following sections for more information:

* [Windows ML on DirectML](#Windows-ML-on-DirectML)
* [ONNX Runtime on DirectML](#ONNX-Runtime-on-DirectML)
* [TensorFlow with DirectML (Preview)](#TensorFlow-with-DirectML-Preview)

## DirectML Samples

DirectML sample code is available under [Samples](./Samples).
* [HelloDirectML](./Samples/HelloDirectML): A minimal "hello world" application that executes a single DirectML operator.
* [DirectMLSuperResolution](./Samples/DirectMLSuperResolution/Samples/ML/DirectMLSuperResolution): A sample that uses DirectML to execute a basic super-resolution model to upscale video from 540p to 1080p in real time.

## Windows ML on DirectML

Windows ML (WinML) is a high-performance, reliable API for deploying hardware-accelerated ML inferences on Windows devices. DirectML provides the GPU backend for Windows ML.

DirectML acceleration can be enabled in Windows ML using the [LearningModelDevice](https://docs.microsoft.com/en-us/uwp/api/windows.ai.machinelearning.learningmodeldevice) with any one of the [DirectX DeviceKinds](https://docs.microsoft.com/en-us/uwp/api/windows.ai.machinelearning.learningmodeldevicekind).

For more information, see [Get Started with Windows ML](https://docs.microsoft.com/en-us/windows/ai/windows-ml/#get-started).

* [Windows Machine Learning Overview (docs.microsoft.com)](https://docs.microsoft.com/en-us/windows/ai/windows-ml/)
* [Windows Machine Learning GitHub](https://github.com/Microsoft/Windows-Machine-Learning)
* [WinMLRunner](https://github.com/Microsoft/Windows-Machine-Learning/tree/master/Tools/WinMLRunner), a tool for executing ONNX models using WinML with DirectML

## ONNX Runtime on DirectML

ONNX Runtime is a cross-platform inferencing and training accelerator compatible with many popular ML/DNN frameworks, including PyTorch, TensorFlow/Keras, scikit-learn, and more.

DirectML is available as an optional *execution provider* for ONNX Runtime that provides hardware acceleration when running on Windows 10.

For more information about getting started, see [Using the DirectML execution provider](https://github.com/microsoft/onnxruntime/blob/master/docs/execution_providers/DirectML-ExecutionProvider.md#using-the-directml-execution-provider).

* [ONNX Runtime homepage](https://aka.ms/onnxruntime)
* [ONNX Runtime GitHub](https://github.com/microsoft/onnxruntime)
* [DirectML Execution Provider readme](https://github.com/microsoft/onnxruntime/blob/master/docs/execution_providers/DirectML-ExecutionProvider.md)

## TensorFlow with DirectML (Preview)

TensorFlow is a popular open source platform for machine learning and is a leading framework for training of machine learning models.

DirectML acceleration for TensorFlow 1.15 is currently available for Public Preview. TensorFlow on DirectML enables training and inference of complex machine learning models on a wide range of DirectX 12-compatible hardware.

TensorFlow on DirectML is supported on both the latest versions of Windows 10 and the [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/about), and is available for download as a PyPI package. For more information about getting started, see [GPU accelerated ML training (docs.microsoft.com)](http://aka.ms/gpuinwsldocs)

* [TensorFlow on DirectML GitHub repo](https://github.com/microsoft/tensorflow-directml)
* [TensorFlow on DirectML samples](./TensorFlow)
* [tensorflow-directml PyPI project](https://pypi.org/project/tensorflow-directml/)
* [TensorFlow GitHub | RFC: TensorFlow on DirectML](https://github.com/tensorflow/community/pull/243)
* [TensorFlow homepage](https://www.tensorflow.org/)

## Feedback

We look forward to hearing from you!

* For TensorFlow with DirectML issues, bugs, and feedback; or for general DirectML issues and feedback, please [file an issue](https://github.com/microsoft/DirectML-Samples/issues) or contact us directly at askdirectml@microsoft.com.

* For Windows ML issues, please file a GitHub issue at [microsoft/Windows-Machine-Learning](https://github.com/Microsoft/Windows-Machine-Learning/issues) or contact us directly at askwindowsml@microsoft.com.

* For ONNX Runtime issues, please file an issue at [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime/issues).

## External Links

### Documentation
[DirectML programming guide](https://docs.microsoft.com/en-us/windows/win32/direct3d12/dml)  
[DirectML API reference](https://docs.microsoft.com/en-us/windows/win32/direct3d12/direct3d-directml-reference)

### More information
[Introducing DirectML (Game Developers Conference '19)](https://www.youtube.com/watch?v=QjQm_wNrvVw)   
[Accelerating GPU Inferencing with DirectML and DirectX 12 (SIGGRAPH '18)](http://on-demand.gputechconf.com/siggraph/2018/video/sig1814-2-adrian-tsai-gpu-inferencing-directml-and-directx-12.html)  
[Windows AI: hardware-accelerated ML on Windows devices (Microsoft Build '20)](https://www.youtube.com/watch?v=-qf2PMuOXWI&feature=youtu.be)  
[Gaming with Windows ML (DirectX Developer Blog)](https://devblogs.microsoft.com/directx/gaming-with-windows-ml/)  
[DirectML at GDC 2019 (DirectX Developer Blog)](https://devblogs.microsoft.com/directx/directml-at-gdc-2019/)  
[DirectX ❤ Linux (DirectX Developer Blog)](https://devblogs.microsoft.com/directx/directx-heart-linux/)

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
