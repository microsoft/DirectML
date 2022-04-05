# YOLO V3 Object Detection

Sample scripts for training and inferencing of the YOLO V3 object detection model using TensorFlow on DirectML.

This is a fork of https://github.com/zzh8829/yolov3-tf2 at commit [c6c42ba](https://github.com/zzh8829/yolov3-tf2/tree/c6c42ba8e9127a0dd6ded2018520754b90d18dae). The original code is Copyright (c) 2019 Zihao Zhang, and is used here under the terms of the MIT License. See [LICENSE](./LICENSE) for more information.

## Setup

First, make sure you have the TensorFlow-DirectML plugin installed. Then install the pip packages required by YOLO V3:

```
pip install -r requirements.txt
```

Next, run the setup script to download all data and pre-trained weights. By default, the script downloads the yolov3 weights. To download the yolov3-tiny weights instead, uncomment line 31.

```bash
python setup.py
```

## Inference

There are two scripts for inference: `detect.py` (for single images) and `detect_video.py` (for video frames). You can also stream frames from a webcam.

```bash
# Image
python detect.py --image <path>

# Image (yolov3-tiny)
python detect.py --weights ./checkpoints/yolov3-tiny.tf --tiny --image <path>

# Video
python detect_video.py --video <path>

# Video with output (yolov3-tiny)
python detect_video.py --video <path> --output <path> --weights ./checkpoints/yolov3-tiny.tf --tiny

# Webcam
python detect_video.py --video 0
```

Press the `Q` key to close the output window and stop the script. Sample images and videos can be found under [`data`](./data).

## Training

It takes quite a bit of time, and a good dataset, to properly train YOLO. You can use the `train_voc.py` helper script to train the model from scratch using the VOC 2012 dataset.

```
python train_voc.py [--epochs <int>] [--batch_size <int>]
```

Note that YOLO V3 can take up quite a bit of memory, so you may need to reduce the batch size to something smaller than the default (4). Try a `--batch_size` of `1` or `2` if you run out of memory.
