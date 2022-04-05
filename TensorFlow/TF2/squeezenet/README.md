# TensorFlow 2.0 DirectML Plugin on SqueezeNet

To run the SqueezeNet training script using the TensorFlow-DirectML plugin, follow the setup instructions below.

## Setup

Run the following commands to set up a conda environment with the correct packages installed to run the squeezenet.py script on the TFDML plugin. These assume that you are running from the directory containing the squeezenet.py script and the TFDML plugin wheel.

```
    conda create --name tfdml_plugin python=3.7
    conda activate tfdml_plugin
    pip install tf-nightly-cpu==2.9.0.dev20220329
    pip install <tfdml plugin>
```

## Sample

The sample SqueezeNet model included in this folder can be run for either training or testing. If only running inference, feel free to use the pre-trained weights in the checkpoints/ folder.

To train:

```
    python squeezenet.py --mode train --tb_profile --cifar10
```

To run inference:

```
    python squeezenet.py --mode test --tb_profile --cifar10
```

Some notes on the script:
    - The `tb_profile` flag in the commands above will profile 20 batches in the first epoch of training. To see the visualizations of the model graph and trace inputs, run `tensorboard --logdir=./train`.
    - The `cifar10` flag specifies a smaller, more compact version of the SqueezeNet model that performs better for the CIFAR-10 dataset as it is a relatively small dataset with few classes. Omitting this flag should work as well.
    - This script downloads and uses the CIFAR-10 dataset. If using a different dataset, you will need to replace get_cifar10_data() in squeezenet.py to download and preprocess that dataset.
    - The model defaults to training for 100 epochs. To train for fewer (1 epoch will be sufficient to capture a TensorBoard trace), add the flag `--num_epochs <number>` to the command above for training.
    - The `--log_device_placement` flag can be used to confirm which operators are being run on DirectML.

## Links

- [Original paper](https://arxiv.org/abs/1602.07360)
- [Original source (Caffe)](https://github.com/forresti/SqueezeNet)
- [Original training data (LSVRC 2012)](http://www.image-net.org/challenges/LSVRC/2012/)
- [Alternative training data (CIFAR-10)](https://www.cs.toronto.edu/~kriz/cifar.html)

Alternative implementations:
- [TensorFlow (Vonclites)](https://github.com/vonclites/squeezenet)
- [TensorFlow (MKOS on CIFAR-10)](https://github.com/mkos/squeezenet)
- [TensorFlow (Tandon-A on Tiny ImageNet)](https://github.com/Tandon-A/SqueezeNet)
- [ONNX](https://github.com/onnx/models/tree/master/vision/classification/squeezenet/squeezenet)