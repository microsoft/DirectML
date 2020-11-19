# Python Binding for DirectML Samples
PyDirectML is an open source Python binding library for DirectML written to facilitate DirectML sample authoring in Python. It provides the following capabilities to Python sample authors:
- Simplified DirectML graph authoring and compilation with operator composition
- Wrapper of DirectML device and resource management
- Binding support through NumPy arrays

## Build and Installation
    python setup.py install

## Usage
In a Python file, import PyDirectML

    import pydirectml

## Samples
DirectML Python sample code is available under the [samples](./samples) folder. These samples require PyDirectML, which can be built and installed to a Python executing environment. 

* [MobileNet](./Python/samples/mobilenet.py): Adapted from the [ONNX MobileNet model](https://github.com/onnx/models/tree/master/vision/classification/mobilenet). MobileNet classifies an image into 1000 different classes. It is highly efficient in speed and size, ideal for mobile applications.
* [MNIST](./Python/samples/mnist.py): Adapted from the [ONNX MNIST model](https://github.com/onnx/models/tree/master/vision/classification/mnist). MNIST predicts handwritten digits using a convolution neural network.
* [SqueezeNet](./Python/samples/squeezenet.py): Based on the [ONNX SqueezeNet model](https://github.com/onnx/models/tree/master/vision/classification/squeezenet). SqueezeNet performs image classification trained on the ImageNet dataset. It is highly efficient and provides results with good accuracy.
* [FNS-Candy](./Python/samples/candy.py): Adapted from the [Windows ML Style Transfer model](https://github.com/microsoft/Windows-Machine-Learning/tree/master/Samples/FNSCandyStyleTransfer) sample, FNS-Candy re-applies specific artistic styles on regular images.
* [Super Resolution](./Python/samples/superres.py): Adapted from the [ONNX Super Resolution model](https://github.com/onnx/models/tree/master/vision/super_resolution/sub_pixel_cnn_2016), Super-Res upscales and sharpens the input images to refine the details and improve image quality.

## Debugging
One of the most effective ways to debug Python files is through [Visual Studio Code](https://code.visualstudio.com/) with [Visual Studio Code Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python). Follow this [instruction](https://code.visualstudio.com/docs/python/debugging) for more details on how to debug Python code in Visual Studio Code.

If mixed-mode debugging across the Python code and the Python binding C++ code becomes necessary, [Visual Studio 2019 with Python](https://visualstudio.microsoft.com/vs/features/python/) provides an excellent support in a single package. You may also want to enable debug information for the Python binding C++ code by editing the `setup.cfg` file and rebuild as follow.

    [build_ext]
    debug=1
