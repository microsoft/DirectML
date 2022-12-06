# maskrcnn Model <!-- omit in toc -->

Sample scripts for training the [Mask R-CNN](https://arxiv.org/abs/1703.06870) model in the [Penn-Fudan Database for Pedestrian Detection and Segmentation](https://www.cis.upenn.edu/~jshi/ped_html/) using PyTorch on DirectML 

These scripts are collected from the tutorial [here](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)

- [Setup](#setup)
- [Prepare Data](#prepare-data)
- [Training](#training)

## Setup
Install the following prerequisites by running the following script from the `root` directory of the DirectML folder:
```
pip install -r pytorch\1.13\objectDetection\maskrcnn\requirements.txt 
```

## Prepare Data

After installing the PyTorch on DirectML package (see [GPU accelerated ML training](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-pytorch-windows)), open a console to the `root` directory and run the setup script to download and convert data:

```
python pytorch\1.13\data\dataset.py
```

Running `dataset.py` should take at least a minute or so, since it downloads the CIFAR-10 dataset. The output of running it should look similar to the following:

```
>python pytorch\1.13\data\dataset.py
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to E:\work\dml\pytorch\1.13\data\cifar-10-python\cifar-10-python.tar.gz
Failed download. Trying https -> http instead. Downloading http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to E:\work\dml\pytorch\1.13\data\cifar-10-python\cifar-10-python.tar.gz
170499072it [00:32, 5250164.09it/s]
Extracting E:\work\dml\pytorch\1.13\data\cifar-10-python\cifar-10-python.tar.gz to E:\work\dml\pytorch\1.13\data\cifar-10-python
```

## Training

A helper script exists to train Mask R-CNN with PennFudanPed data:

```
cd pytorch\1.13\objectdetection\maskrcnn
python .\maskrcnn.py
```

The first few lines of output should look similar to the following (exact numbers may change):
```
>python .\maskrcnn.py
python .\maskrcnn.py
Epoch: [0]  [ 0/60]  eta: 0:38:26  lr: 0.000090  loss: 2.9777 (2.9777)  loss_classifier: 0.7217 (0.7217)  loss_box_reg: 0.0754 (0.0754)  loss_mask: 1.6228 (1.6228)  loss_objectness: 0.4175 (0.4175)  loss_rpn_box_reg: 0.1404 (0.1404)  time: 38.4439  data: 1.0955
Epoch: [0]  [10/60]  eta: 0:29:44  lr: 0.000936  loss: 2.4268 (2.4919)  loss_classifier: 0.4056 (0.4158)  loss_box_reg: 0.1691 (0.3631)  loss_mask: 1.1679 (1.1600)  loss_objectness: 0.1162 (0.3120)  loss_rpn_box_reg: 0.1257 (0.2410)  time: 35.6972  data: 0.1034
Epoch: [0]  [20/60]  eta: 0:23:14  lr: 0.001783  loss: 1.2172 (1.6717)  loss_classifier: 0.0669 (0.2410)  loss_box_reg: 0.1331 (0.2466)  loss_mask: 0.5935 (0.8376)  loss_objectness: 0.0565 (0.1873)  loss_rpn_box_reg: 0.0574 (0.1593)  time: 34.6860  data: 0.0042
```
