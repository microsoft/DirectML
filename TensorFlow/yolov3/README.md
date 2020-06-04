# YOLO V3 Object Detection

**This is a fork of https://github.com/zzh8829/yolov3-tf2 at commit 65294d5dc1794b325db5a37b2ed02773ca5bf839.** Here are a few of the changes in this fork:

- Setup
  - Add a simpler `setup.py` to download and prepare all inference and training data.
- Inference (`detect.py` and `detect_video.py`)
  - Render image with predictions on screen instead of writing to disk (by default).
  - Add non-eager execution mode, which is used by default. This tends to be noticeably faster than eager mode, but an `--eager` switch is added for completeness.
  - Add tracing of frames, which forces non-eager execution.
  - Outline objects with class-specific colors set in `data/colors.json`. Random colors are used for unrecognized classes.
  - Add options for headless execution (useful for WSL & profiling) and frame limit in videos.
- Training
  - Add missing initializers (for non-eager mode) and enable eager execution (eager mode).
  - Add a toy dataset for quick profiling without downloading a massive dataset.
  - Add some wrapper scripts for training (`train_voc.py` and `train_toy.py`).
  - Add tracing of batches, which requires non-eager execution.

## Setup

First, make sure you have the required pip packages:

```
pip install -r requirements.txt
```

Next, run the setup script to download all data and pre-trained weights. This is necessary if you want to train with real data or run inference with pre-trained weights. If you only want to profile the model, you can use the included toy dataset (see training section below).

```bash
python setup.py
```

## Inference

There are two scripts for inferece: `detect.py` (for single images) and `detect_video.py` (for video frames). You can also stream frames from a webcam.

```bash
# Image
python detect.py --image data/mexico.jpg

# Video
python detect_video.py --video data/ski.mp4

# Webcam
python detect_video.py --video 0
```

Press the `Q` key to close the output window and stop the script.

<!-- If you want to use YOLOV3-Tiny, just append `--weights checkpoints/yolov3-tiny.tf --tiny` to the above command lines.  -->

There are a few additional parameters you may find useful:

- `--output <path>` : output the image/video with labels and boxes applied (e.g. `--output processed.avi`)
- `--headless` : dont't render output on screen, which is useful for tracing and WSL
- `--trace` : emit chrome traces for each image/frame
- `--max_frames <int>` : cap on the number of video frames to process (detect_video.py only)

The inference scripts support tracing. For example, to trace first 10 frames of a video without rendering it on screen:

```
python detect_video.py --video data/ski.mp4 --trace --max_frames 10 --headless
```

## Training

It takes quite a bit of time, and a good dataset, to properly train YOLO. For quick validation of the model, it's best to use the `train_toy.py` script. This script simply uses a single image as the dataset.

```
python train_toy.py [--trace]
```

Running the above script goes through 4 epochs with batch size 1. You should see output similar to the following (the loss values will be meaningless, as this is only for tracing and testing):

```
Epoch 00001: saving model to checkpoints/yolov3_train_1.tf
1/1 [==============================] - 8s 8s/step - loss: 7392.9839 - yolo_output_0_loss: 332.3820 - yolo_output_1_loss: 1393.9207 - yolo_output_2_loss: 5654.6016 - val_loss: 137597376.0000 - val_yolo_output_0_loss: 10925322.0000 - val_yolo_output_1_loss: 126672008.0000 - val_yolo_output_2_loss: 26.5903
Epoch 2/4

Epoch 00002: saving model to checkpoints/yolov3_train_2.tf
1/1 [==============================] - 4s 4s/step - loss: 137597264.0000 - yolo_output_0_loss: 10925311.0000 - yolo_output_1_loss: 126671912.0000 - yolo_output_2_loss: 26.5903 - val_loss: 6809.0308 - val_yolo_output_0_loss: 89.9505 - val_yolo_output_1_loss: 1258.9493 - val_yolo_output_2_loss: 5448.0376
Epoch 3/4

Epoch 00003: saving model to checkpoints/yolov3_train_3.tf
1/1 [==============================] - 4s 4s/step - loss: 6809.0308 - yolo_output_0_loss: 89.9505 - yolo_output_1_loss: 1258.9493 - yolo_output_2_loss: 5448.0376 - val_loss: 6439.5229 - val_yolo_output_0_loss: 111.3732 - val_yolo_output_1_loss: 1302.4730 - val_yolo_output_2_loss: 5013.5625
Epoch 4/4

Epoch 00004: saving model to checkpoints/yolov3_train_4.tf
1/1 [==============================] - 4s 4s/step - loss: 6439.5229 - yolo_output_0_loss: 111.3732 - yolo_output_1_loss: 1302.4730 - yolo_output_2_loss: 5013.5625 - val_loss: 4428.4683 - val_yolo_output_0_loss: 58.6571 - val_yolo_output_1_loss: 1214.9315 - val_yolo_output_2_loss: 3142.7344
```

To train from scratch with the VOC 2012 dataset, use the `train_voc.py` helper:

```
python train_voc.py [--epochs <int>] [--batch_size <int>] [--trace]
```

**NOTE**: YOLO can take up quite a bit of memory, so you may need to reduce the batch size to something smaller than the default (8). Try 4 if you run out of memory.
