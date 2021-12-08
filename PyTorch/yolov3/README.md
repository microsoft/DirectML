# YOLO V3 Object Detection

Sample scripts for training and inferencing of the YOLO V3 object detection model using Pytorch on DirectML.

This is a fork of https://github.com/ultralytics/yolov3 at commit [1be3170](https://github.com/ultralytics/yolov3/commit/1be31704c9c690929e4f6e6d950f40755ef2dcdc). The original code is here under the terms of the GNU GENERAL PUBLIC LICENSE. See [LICENSE](./LICENSE) for more information.

- [YOLO V3 Object Detection](#yolo-v3-object-detection)
  - [Setup](#setup)
  - [Training](#training)
  - [Testing](#testing)
  - [Predict](#predict)
  - [Links](#links)

## Setup
Install the following prerequisites:
```
pip install -r pytorch\yolov3\requirements.txt 
```

## Training

A helper script exists to train yolov3 with default data, batch size, and so on:

```
python train.py --batch-size 4 --device dml
```

The first few lines of output should look similar to the following (exact numbers may change):
We choose (3) Don't visualize my results for wandb to focus on trainning.
```
>(pytorch-dml) PS **\DirectML\PyTorch\yolov3> python .\train.py --batch-size=4 --device dml
Namespace(adam=False, artifact_alias='latest', batch_size=4, bbox_interval=-1, bucket='', cache_images=False, cfg='', data='data/coco128.yaml', device='dml', entity=None, epochs=300, evolve=False, exist_ok=False, global_rank=-1, hyp='data/hyp.scratch.yaml', image_weights=False, img_size=[640, 640], label_smoothing=0.0, linear_lr=False, local_rank=-1, multi_scale=False, name='exp', noautoanchor=False, nosave=False, notest=False, project='runs/train', quad=False, rect=False, resume=False, save_dir='runs\\train\\exp4', save_period=-1, single_cls=False, sync_bn=False, total_batch_size=4, upload_dataset=False, weights='yolov3.pt', workers=8, world_size=1)
tensorboard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
hyperparameters: lr0=0.01, lrf=0.2, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
←[34m←[1mwandb←[0m: Enter your choice: 3
```

By default, the script will run for 50 epochs with a batch size of 32 and print the accuracy after every 100 batches. The training script can be run multiple times and saves progress after each epoch (by default).

The accuracy should increase over time.

You can inspect `train.py` (and the real script, `pytorch/classification/train_classification.py`) to see the command line it is invoking or adjust some of the parameters. Increasing the batch size will, in general, improve the accuracy. 

You can save the model for testing by passing in the --save_model flag. This will cause checkpoints to be saved to the pytorch\checkpoints\<device>\<model>\checkpoint.pth path.

```
python pytorch\resnet50\train.py --save_model
```


## Testing

Once the model is trained and saved we can now test the model using the following steps. The test script will use the latest trained model from the checkpoints folder.

```
python pytorch\resnet50\test.py
```

You should see the result such as this:

```
>python pytorch\resnet50\test.py
Loading the testing dataset from: E:\work\dml\PyTorch\data\cifar-10-python
        Test data X [N, C, H, W]:
                shape=torch.Size([32, 3, 224, 224]),
                dtype=torch.float32
        Test data Y:
                shape=torch.Size([32]),
                dtype=torch.int64
Finished moving resnet50 to device: dml in 0.6159994602203369s.
current highest_accuracy:  0.10559999942779541
Test Error:
 Accuracy: 10.0%, Avg loss: 2.321213
```
## Predict

Once the model is trained and saved we can now run the prediction using the following steps. The predict script will use that latest trained model from the checkpoints folder.

```
python pytorch\squeezenet\predict.py --image E:\a.jpeg
```

You should see the result such as this:

```
E:\work\dml>python pytorch\squeezenet\predict.py --image E:\a.jpeg
hammerhead 0.35642221570014954
stingray 0.34619468450546265
electric ray 0.09593362361192703
cock 0.07319413870573044
great white shark 0.06555310636758804
```

## Links

- [Original training data (LSVRC 2012)](http://www.image-net.org/challenges/LSVRC/2012/)
- [Alternative training data (CIFAR-10)](https://www.cs.toronto.edu/~kriz/cifar.html)

Alternative implementations:
- [ONNX](https://github.com/onnx/models/tree/master/vision/classification/resnet)