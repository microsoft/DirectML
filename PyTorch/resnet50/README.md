# Resnet50 Model <!-- omit in toc -->

Sample scripts for training the Resnet50 model using PyTorch on DirectML.

These scripts were forked from https://github.com/pytorch/benchmark. The original code is Copyright (c) 2019, pytorch, and is used here under the terms of the BSD 3-Clause License. See [LICENSE](https://github.com/pytorch/benchmark/blob/main/LICENSE) for more information.

The original paper can be found at: https://arxiv.org/abs/1602.07360

- [Setup](#setup)
- [Prepare Data](#prepare-data)
- [Training](#training)
- [Testing](#testing)
- [Tracing](#tracing)
- [Links](#links)

## Setup
Install the following prerequisites:
```
pip install -r pytorch\resnet50\requirements.txt 
```

## Prepare Data

After installing the PyTorch on DirectML package (see [GPU accelerated ML training](..\readme.md)), open a console to the `root` directory and run the setup script to download and convert data:

```
python pytorch\data\cifar.py
```

Running `setup.py` should take at least a minute or so, since it downloads the CIFAR-10 dataset. The output of running it should look similar to the following:

```
>python pytorch\data\cifar.py
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to E:\work\dml\PyTorch\data\cifar-10-python\cifar-10-python.tar.gz
Failed download. Trying https -> http instead. Downloading http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to E:\work\dml\PyTorch\data\cifar-10-python\cifar-10-python.tar.gz
170499072it [00:32, 5250164.09it/s]
Extracting E:\work\dml\PyTorch\data\cifar-10-python\cifar-10-python.tar.gz to E:\work\dml\PyTorch\data\cifar-10-python
```

## Training

A helper script exists to train Resnet50 with default data, batch size, and so on:

```
python pytorch\resnet50\train.py
```

The first few lines of output should look similar to the following (exact numbers may change):
```
>python pytorch\resnet50\train.py
Loading the training dataset from: E:\work\dml\PyTorch\data\cifar-10-python
        Train data X [N, C, H, W]:
                shape=torch.Size([32, 3, 224, 224]),
                dtype=torch.float32
        Train data Y:
                shape=torch.Size([32]),
                dtype=torch.int64
Loading the testing dataset from: E:\work\dml\PyTorch\data\cifar-10-python
        Test data X [N, C, H, W]:
                shape=torch.Size([32, 3, 224, 224]),
                dtype=torch.float32
        Test data Y:
                shape=torch.Size([32]),
                dtype=torch.int64
Finished moving resnet50 to device: dml in 0.2560007572174072s.
Epoch 1
-------------------------------
loss: 2.309573  [    0/50000]
loss: 2.324099  [ 3200/50000]
loss: 2.297763  [ 6400/50000]
loss: 2.292575  [ 9600/50000]
loss: 2.251738  [12800/50000]
loss: 2.183397  [16000/50000]
loss: 2.130508  [19200/50000]
loss: 2.000042  [22400/50000]
loss: 2.183213  [25600/50000]
loss: 2.250935  [28800/50000]
loss: 1.730087  [32000/50000]
loss: 1.999480  [35200/50000]
loss: 1.865684  [38400/50000]
loss: 2.058377  [41600/50000]
loss: 2.059475  [44800/50000]
loss: 2.279521  [48000/50000]
current highest_accuracy:  0.2856000065803528
Test Error:
 Accuracy: 28.6%, Avg loss: 1.862064
```

By default, the script will run for 50 epochs with a batch size of 32 and print the accuracy after every 100 batches. The training script can be run multiple times and saves progress after each epoch (by default).

The accuracy should increase over time.

You can inspect `train.py` (and the real script, `pytorch/classification/train_classification.py`) to see the command line it is invoking or adjust some of the parameters. Increasing the batch size will, in general, improve the accuracy. 

Once you're finished with training (at least one checkpoint must exist), you can save the model for testing.

```
python pytorch\resnet50\train.py --save_model
```

## Testing

Once the model is trained and saved we can now run the prediction using the following steps. Feel free to change the input image from the test set. Make sure that the data_format layout matches the same layout that the model is trained with.

```
python src/predict_squeezenet.py --model_dir data --data_format NCHW --image data/cifar10_images/test/ship/0001.png
```

You should see the result such as this:

```
data/cifar10_images/test/ship/0001.png: predicted ship
```

## Tracing

It may be useful to get a trace during training or evaluation.

```
python pytorch\resnet50\test.py --trace
python pytorch\resnet50\train.py --trace
```

With default settings, you'll see output like the following:

```

```

## Links

- [Original training data (LSVRC 2012)](http://www.image-net.org/challenges/LSVRC/2012/)
- [Alternative training data (CIFAR-10)](https://www.cs.toronto.edu/~kriz/cifar.html)

Alternative implementations:
- [ONNX](https://github.com/onnx/models/tree/master/vision/classification/resnet)