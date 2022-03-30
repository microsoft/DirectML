# YOLO V3 Object Detection

Sample scripts for training and inferencing of the YOLO V3 object detection model using TensorFlow on DirectML.

This is a fork of https://github.com/zzh8829/yolov3-tf2 at commit [65294d5](https://github.com/zzh8829/yolov3-tf2/tree/65294d5dc1794b325db5a37b2ed02773ca5bf839). The original code is Copyright (c) 2019 Zihao Zhang, and is used here under the terms of the MIT License. See [LICENSE](./LICENSE) for more information.

## Setup

First, make sure you have the TensorFlow on DirectML package installed (see [GPU accelerated ML training](http://aka.ms/gpuinwsldocs)). Then install the pip packages required by YOLO V3:

```
pip install -r requirements.txt
```

Next, run the setup script to download all data and pre-trained weights.

```bash
python setup.py
```

## Inference

There are two scripts for inference: `detect.py` (for single images) and `detect_video.py` (for video frames). You can also stream frames from a webcam.

```bash
# Image
python detect.py --image <path>

# Video
python detect_video.py --video <path>

# Webcam
python detect_video.py --video 0
```

Press the `Q` key to close the output window and stop the script. Sample images and videos can be found under [`data`](./data).

There are a few additional parameters you may find useful:

- `--output <path>` : output the image/video with labels and boxes applied (e.g. `--output processed.avi`)
- `--headless` : don't render output on screen, which is useful for tracing and WSL
- `--trace` : emit chrome traces for each image/frame
- `--max_frames <int>` : cap on the number of video frames to process (detect_video.py only)

The inference scripts support tracing. For example, to trace first 10 frames of a video without rendering it on screen:

```
python detect_video.py --video data/grca-trainmix_1280x720.mp4 --trace --max_frames 10 --headless
```

## Training

It takes quite a bit of time, and a good dataset, to properly train YOLO. You can use the `train_voc.py` helper script to train the model from scratch using the VOC 2012 dataset.

```
python train_voc.py [--epochs <int>] [--batch_size <int>] [--trace]
```

Note that YOLO V3 can take up quite a bit of memory, so you may need to reduce the batch size to something smaller than the default (4). Try a `--batch_size` of `1` or `2` if you run out of memory.
