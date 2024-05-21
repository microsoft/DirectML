# SqueezeNet Model <!-- omit in toc -->

Sample scripts for training the SqueezeNet model using PyTorch on DirectML.

These scripts were forked from https://github.com/pytorch/benchmark. The original code is Copyright (c) 2019, pytorch, and is used here under the terms of the BSD 3-Clause License. See [LICENSE](https://github.com/pytorch/benchmark/blob/main/LICENSE) for more information.

The original paper can be found at: https://arxiv.org/abs/1602.07360

- [Setup](#setup)
- [Prepare Data](#prepare-data)
- [Training](#training)
- [Testing](#testing)
- [Predict](#predict)
- [Tracing](#tracing)
- [External Links](#external-links)

## Setup
Install the following prerequisites:
```
pip install -r pytorch\cv\squeezenet\requirements.txt 
```

## Prepare Data

After installing the PyTorch on DirectML package (see [GPU accelerated ML training](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-pytorch-windows)), open a console to the `root` directory and run the setup script to download and convert data:

```
python pytorch\cv\data\dataset.py
```

Running `setup.py` should take at least a minute or so, since it downloads the CIFAR-10 dataset. The output of running it should look similar to the following:

```
>python pytorch\cv\data\dataset.py
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to E:\work\dml\pytorch\cv\data\cifar-10-python\cifar-10-python.tar.gz
Failed download. Trying https -> http instead. Downloading http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to E:\work\dml\pytorch\cv\data\cifar-10-python\cifar-10-python.tar.gz
170499072it [00:32, 5250164.09it/s]
Extracting E:\work\dml\pytorch\cv\data\cifar-10-python\cifar-10-python.tar.gz to E:\work\dml\pytorch\cv\data\cifar-10-python
```

## Training

A helper script exists to train SqueezeNet with default data, batch size, and so on:

```
python pytorch\cv\squeezenet\train.py
```

The first few lines of output should look similar to the following (exact numbers may change):
```
>python pytorch\cv\squeezenet\train.py
Loading the training dataset from: E:\work\dml\pytorch\cv\data\cifar-10-python
        Train data X [N, C, H, W]:
                shape=torch.Size([32, 3, 224, 224]),
                dtype=torch.float32
        Train data Y:
                shape=torch.Size([32]),
                dtype=torch.int64
Loading the testing dataset from: E:\work\dml\pytorch\cv\data\cifar-10-python
        Test data X [N, C, H, W]:
                shape=torch.Size([32, 3, 224, 224]),
                dtype=torch.float32
        Test data Y:
                shape=torch.Size([32]),
                dtype=torch.int64
Finished moving squeezenet1_1 to device: dml in 0.2560007572174072s.
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

By default, the script will run for 50 epochs with a batch size of 32 and print the accuracy after every 100 batches. The training script can be run multiple times and saves progress after each epoch (by default). The accuracy should increase over time.

> When discrete memory or shared GPU memory is insufficient consider running the same scripts with a smaller batch size (use the --batch_size argument). For example:

```
python pytorch\cv\resnet50\train.py --batch_size 8
```

You can inspect `train.py` (and the real script, `pytorch\cv\classification\train_classification.py`) to see the command line it is invoking or adjust some of the parameters.

You can save the model for testing by passing in the --save_model flag. This will cause checkpoints to be saved to the `pytorch\cv\checkpoints\<device>\<model>\checkpoint.pth` path.

```
python pytorch\cv\resnet50\train.py --save_model
```

## Testing

Once the model is trained and saved we can now test the model using the following steps. The test script will use the latest trained model from the checkpoints folder.

```
python pytorch\cv\squeezenet\test.py
```

You should see the result such as this:

```
>python pytorch\cv\squeezenet\test.py
Loading the testing dataset from: E:\work\dml\pytorch\cv\data\cifar-10-python
        Test data X [N, C, H, W]:
                shape=torch.Size([32, 3, 224, 224]),
                dtype=torch.float32
        Test data Y:
                shape=torch.Size([32]),
                dtype=torch.int64
Finished moving squeezenet1_1 to device: dml in 0.22499728202819824s.
current highest_accuracy:  0.10000000149011612
Test Error:
 Accuracy: 10.0%, Avg loss: 2.321213
```
## Predict

Once the model is trained and saved we can now run the prediction using the following steps. The predict script will use that latest trained model from the checkpoints folder.

```
python pytorch\cv\squeezenet\predict.py --image E:\a.jpeg
```

You should see the result such as this:

```
E:\work\dml>python pytorch\cv\squeezenet\predict.py --image E:\a.jpeg
hammerhead 0.35642221570014954
stingray 0.34619468450546265
electric ray 0.09593362361192703
cock 0.07319413870573044
great white shark 0.06555310636758804
```

## Tracing

It may be useful to get a trace during training or evaluation.

```
python pytorch\cv\squeezenet\test.py --trace True
python pytorch\cv\squeezenet\train.py --trace True
```

With default settings, you'll see output like the following:

```
>python pytorch\cv\squeezenet\train.py --trace True
Loading the training dataset from: E:\work\dml\pytorch\cv\data\cifar-10-python
        Train data X [N, C, H, W]:
                shape=torch.Size([1, 3, 224, 224]),
                dtype=torch.float32
        Train data Y:
                shape=torch.Size([1]),
                dtype=torch.int64
Loading the testing dataset from: E:\work\dml\pytorch\cv\data\cifar-10-python
        Test data X [N, C, H, W]:
                shape=torch.Size([1, 3, 224, 224]),
                dtype=torch.float32
        Test data Y:
                shape=torch.Size([1]),
                dtype=torch.int64
Finished moving squeezenet1_1 to device: dml in 0.2282116413116455s.
Epoch 1
-------------------------------
------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                      Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls
------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                           model_inference        33.98%     244.942ms        67.93%     489.574ms     489.574ms          -4 b         -20 b             1
                        ThnnConv2DBackward         0.06%     435.700us        21.73%     156.616ms       6.024ms           0 b           0 b            26
                aten::thnn_conv2d_backward        21.52%     155.095ms        21.67%     156.180ms       6.007ms           0 b           0 b            26
                              aten::conv2d         0.15%       1.070ms        13.12%      94.566ms       3.637ms           0 b           0 b            26
                         aten::convolution         0.12%     877.800us        12.97%      93.496ms       3.596ms           0 b           0 b            26
                        aten::_convolution         0.14%     975.500us        12.85%      92.618ms       3.562ms           0 b           0 b            26
                aten::_convolution_nogroup         0.12%     889.600us        12.71%      91.643ms       3.525ms           0 b           0 b            26
                         aten::thnn_conv2d         0.12%     858.900us        12.59%      90.753ms       3.491ms           0 b           0 b            26
                 aten::thnn_conv2d_forward        12.01%      86.566ms        12.47%      89.894ms       3.457ms           0 b           0 b            26
                   Optimizer.step#SGD.step         0.52%       3.769ms        10.38%      74.808ms      74.808ms          -4 b         -20 b             1
                                 aten::add         4.57%      32.967ms         4.57%      32.967ms     633.988us           0 b           0 b            52
                             ReluBackward1         0.03%     219.000us         4.01%      28.888ms       1.111ms           0 b           0 b            26
                  aten::threshold_backward         3.98%      28.669ms         3.98%      28.669ms       1.103ms           0 b           0 b            26
                       aten::empty_strided         3.82%      27.552ms         3.82%      27.552ms     257.492us           4 b           4 b           107
    struct torch::autograd::AccumulateGrad         0.13%     905.400us         3.19%      22.985ms     442.012us           0 b           0 b            52
                               aten::clone         0.52%       3.726ms         2.79%      20.118ms     386.875us           0 b           0 b            52
                                aten::add_         2.23%      16.077ms         2.23%      16.077ms     309.167us           0 b           0 b            52
                   aten::new_empty_strided         0.06%     450.100us         2.02%      14.575ms     280.285us           0 b           0 b            52
                         aten::log_softmax         0.00%      31.800us         1.95%      14.039ms      14.039ms           0 b           0 b             1
                        aten::_log_softmax         1.94%      14.007ms         1.94%      14.007ms      14.007ms           0 b           0 b             1
                               aten::copy_         1.59%      11.450ms         1.59%      11.450ms     107.012us           0 b           0 b           107
                            aten::nll_loss         0.01%      51.200us         1.52%      10.988ms      10.988ms           0 b           0 b             1
                                 aten::cat         0.06%     439.200us         1.52%      10.964ms       1.370ms           0 b           0 b             8
                    aten::nll_loss_forward         1.50%      10.779ms         1.52%      10.937ms      10.937ms           0 b           0 b             1
                             aten::dropout         0.01%      97.400us         1.50%      10.809ms      10.809ms           0 b           0 b             1
                                aten::_cat         1.46%      10.525ms         1.46%      10.525ms       1.316ms           0 b           0 b             8
                          aten::max_pool2d         0.02%     143.300us         1.10%       7.919ms       2.640ms           0 b           0 b             3
             aten::max_pool2d_with_indices         1.08%       7.776ms         1.08%       7.776ms       2.592ms           0 b           0 b             3
                               aten::relu_         0.98%       7.045ms         0.98%       7.045ms     270.969us           0 b           0 b            26
              MaxPool2DWithIndicesBackward         0.01%      55.600us         0.87%       6.302ms       2.101ms           0 b           0 b             3
    aten::max_pool2d_with_indices_backward         0.87%       6.246ms         0.87%       6.246ms       2.082ms           0 b           0 b             3
                 aten::adaptive_avg_pool2d         0.01%      43.100us         0.85%       6.109ms       6.109ms           0 b           0 b             1
                aten::_adaptive_avg_pool2d         0.84%       6.066ms         0.84%       6.066ms       6.066ms           0 b           0 b             1
                          aten::as_strided         0.82%       5.932ms         0.82%       5.932ms      26.249us           0 b           0 b           226
                                aten::div_         0.57%       4.096ms         0.64%       4.628ms       4.628ms           0 b          -4 b             1
                        LogSoftmaxBackward         0.00%      21.700us         0.64%       4.585ms       4.585ms           0 b           0 b             1
                                 aten::mul         0.64%       4.579ms         0.64%       4.579ms       2.290ms           0 b           0 b             2
          aten::_log_softmax_backward_data         0.63%       4.563ms         0.63%       4.563ms       4.563ms           0 b           0 b             1
                 AdaptiveAvgPool2DBackward         0.00%      13.000us         0.62%       4.496ms       4.496ms           0 b           0 b             1
                               CatBackward         0.01%      63.400us         0.62%       4.486ms     560.712us           0 b           0 b             8
       aten::_adaptive_avg_pool2d_backward         0.62%       4.483ms         0.62%       4.483ms       4.483ms           0 b           0 b             1
                           aten::ones_like         0.00%      29.700us         0.62%       4.478ms       4.478ms           0 b           0 b             1
                              aten::narrow         0.01%      54.100us         0.61%       4.422ms     276.394us           0 b           0 b            16
                               aten::slice         0.02%     155.500us         0.61%       4.368ms     273.013us           0 b           0 b            16
                               aten::fill_         0.57%       4.120ms         0.57%       4.120ms       4.120ms           0 b           0 b             1
                               aten::empty         0.43%       3.080ms         0.43%       3.080ms      16.296us     338.06 Kb     338.06 Kb           189
                          aten::bernoulli_         0.17%       1.233ms         0.37%       2.636ms       1.318ms           0 b    -338.00 Kb             2
         Optimizer.zero_grad#SGD.zero_grad         0.06%     400.600us         0.36%       2.623ms       2.623ms          -4 b         -20 b             1
                               aten::zero_         0.31%       2.250ms         0.31%       2.250ms      40.902us           0 b           0 b            55
                           NllLossBackward         0.01%      48.700us         0.30%       2.180ms       2.180ms           0 b           0 b             1
                   aten::nll_loss_backward         0.29%       2.118ms         0.30%       2.132ms       2.132ms           0 b           0 b             1
                              aten::detach         0.14%       1.012ms         0.26%       1.864ms      35.854us           0 b           0 b            52
                          aten::empty_like         0.02%     110.800us         0.13%     914.800us     304.933us     338.00 Kb           0 b             3
                                    detach         0.12%     852.700us         0.12%     852.700us      16.398us           0 b           0 b            52
                              MulBackward0         0.00%      16.700us         0.07%     539.700us     539.700us           0 b           0 b             1
                                  aten::to         0.01%      57.700us         0.07%     532.200us     266.100us           4 b           0 b             2
                           aten::transpose         0.04%     277.300us         0.06%     413.300us       5.299us           0 b           0 b            78
                               aten::zeros         0.02%     117.300us         0.03%     249.100us      83.033us          12 b           0 b             3
                             aten::reshape         0.01%      57.100us         0.01%      96.300us      32.100us           0 b           0 b             3
                             aten::flatten         0.01%      54.100us         0.01%      95.300us      95.300us           0 b           0 b             1
                             aten::squeeze         0.01%      60.300us         0.01%      93.900us      46.950us           0 b           0 b             2
                                aten::view         0.01%      80.400us         0.01%      80.400us      20.100us           0 b           0 b             4
                              ViewBackward         0.00%       7.400us         0.00%      25.400us      25.400us           0 b           0 b             1
                                aten::conj         0.00%       7.500us         0.00%       7.500us       7.500us           0 b           0 b             1
------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 720.753ms

Done! with highest_accuracy:  0

```

## External Links

- [Original paper](https://arxiv.org/abs/1602.07360)
- [Original training data (LSVRC 2012)](http://www.image-net.org/challenges/LSVRC/2012/)
- [Alternative training data (CIFAR-10)](https://www.cs.toronto.edu/~kriz/cifar.html)

Alternative implementations:
- [ONNX](https://github.com/onnx/models/tree/master/vision/classification/squeezenet)