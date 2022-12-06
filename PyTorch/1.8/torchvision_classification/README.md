# Resnet50 Model <!-- omit in toc -->\\

Sample scripts for training the [torchvision(v0.9.0) classification models](https://pytorch.org/vision/0.9/models.html#classification) using PyTorch on DirectML.

- [Setup](#setup)
- [Prepare Data](#prepare-data)
- [Training](#training)
- [Testing](#testing)
- [Predict](#predict)
- [Links](#links)

## Setup
Install the following prerequisites:
```
pip install -r pytorch\1.8\torchvision_classification\requirements.txt 
```

## Prepare Data

After installing the PyTorch on DirectML package (see [GPU accelerated ML training](https://docs.microsoft.com/en-us/windows/ai/directml/gpu-pytorch-windows)), open a console to the `root` directory and run the setup script to download and convert data:

```
python pytorch\1.8\data\dataset.py
```

Running `setup.py` should take at least a minute or so, since it downloads the CIFAR-10 dataset and PennFudanPed dataset. The output of running it should look similar to the following:

```
>python pytorch\1.8\data\dataset.py
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to E:\work\DirectML\Pytorch\1.8\data\cifar-10-python.tar.gz
Failed download. Trying https -> http instead. Downloading http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to E:\work\DirectML\Pytorch\1.8\data\cifar-10-python.tar.gz
170499072it [00:17, 9709154.90it/s]
Extracting E:\work\DirectML\Pytorch\1.8\data\cifar-10-python.tar.gz to E:\work\DirectML\Pytorch\1.8\data
Downloading PennFundaPed dataset

100% [........................................................................] 53723336 / 53723336
Extracted PennFundaPed dataset
```

## Training

A helper script exists to train classification models with default data, batch size, and so on:

```
python pytorch\1.8\torchvision_classification\train.py --model resnet18
```

model names from list below can be used to train:
- resnet18
- alexnet
- vgg16
- squeezenet1_0
- densenet161
- inception_v3
- googlenet
- shufflenet_v2_x1_0
- mobilenet_v2
- mobilenet_v3_large
- mobilenet_v3_small
- resnext50_32x4d
- wide_resnet50_2
- mnasnet1_0

The first few lines of output should look similar to the following (exact numbers may change):
```
> python pytorch\1.8\torchvision_classification\train.py --model resnet18
Namespace(batch_size=32, device='dml', epochs=50, learning_rate=0.001, model='resnet18', momentum=0.9, path='cifar-10-python', save_model=False, trace=False, weight_decay=0.0001)
Loading the training dataset from: F:\DirectML\Pytorch\1.8\data\cifar-10-python
        Train data X [N, C, H, W]:
                shape=torch.Size([32, 3, 224, 224]),
                dtype=torch.float32
        Train data Y:
                shape=torch.Size([32]),
                dtype=torch.int64
Loading the testing dataset from: F:\DirectML\Pytorch\1.8\data\cifar-10-python
        Test data X [N, C, H, W]:
                shape=torch.Size([32, 3, 224, 224]),
                dtype=torch.float32
        Test data Y:
                shape=torch.Size([32]),
                dtype=torch.int64
Finished moving resnet18 to device: dml in 0.0s.
Epoch 1
-------------------------------
loss: 2.253009  [ 3168/50000] in 15.150734s
loss: 2.159135  [ 6368/50000] in 14.394703s
loss: 1.814574  [ 9568/50000] in 13.828154s
loss: 2.250808  [12768/50000] in 13.965441s
```

By default, the script will run for 50 epochs with a batch size of 32 and print the accuracy after every 100 batches. The training script can be run multiple times and saves progress after each epoch (by default). The accuracy should increase over time.

> When discrete memory or shared GPU memory is insufficient consider running the same scripts with a smaller batch size (use the --batch_size argument). For example:

```
python pytorch\1.8\torchvision_classification\train.py --model resnet18 --batch_size 8
```

You can inspect `train.py` (and the real script, `pytorch/1.8/classification/train_classification.py`) to see the command line it is invoking or adjust some of the parameters. Increasing the batch size will, in general, improve the accuracy. 

You can save the model for testing by passing in the --save_model flag. This will cause checkpoints to be saved to the pytorch\1.8\checkpoints\<device>\<model>\checkpoint.pth path.

```
python pytorch\1.8\torchvision_classification\train.py --save_model
```


## Testing

Once the model is trained and saved we can now test the model using the following steps. The test script will use the latest trained model from the checkpoints folder.

```
python pytorch\1.8\torchvision_classification\test.py --model resnet18
```

You should see the result such as this:

```
PS F:\DirectML> python pytorch\1.8\torchvision_classification\test.py --model resnet18
Namespace(batch_size=32, device='dml', model='resnet18', path='cifar-10-python', trace=False)
Loading the testing dataset from: F:\DirectML\Pytorch\1.8\data\cifar-10-python
        Test data X [N, C, H, W]:
                shape=torch.Size([32, 3, 224, 224]),
                dtype=torch.float32
        Test data Y:
                shape=torch.Size([32]),
                dtype=torch.int64
Finished moving resnet50 to device: dml in 1.0158095359802246s.
current highest_accuracy:  0.09629999846220016
Test Error:
 Accuracy: 9.6%, Avg loss: 8.577250
```

## Predict

Once the model is trained and saved we can now run the prediction using the following steps. The predict script will use that latest trained model from the checkpoints folder.

```
python pytorch\1.8\torchvision_classification\predict.py --model resnet18 --image E:\a.jpeg
```

You should see the result such as this:

```
E:\work\dml>python pytorch\1.8\torchvision_classification\predict.py --model resnet18 --image E:\a.jpeg
cock 0.10412269830703735
electric ray 0.1026773527264595
tench 0.10185252875089645
great white shark 0.10128137469291687
hammerhead 0.09998250752687454
```



## Links

- [Original training data (LSVRC 2012)](http://www.image-net.org/challenges/LSVRC/2012/)
- [Alternative training data (CIFAR-10)](https://www.cs.toronto.edu/~kriz/cifar.html)
