#!/usr/bin/env python
import subprocess
import argparse
import os
import sys
import re
from tensorflow.train import latest_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32)
parser.add_argument("--data_format", choices=("NCHW", "NHWC"), default="NCHW")
parser.add_argument("--max_train_steps", default=1801, type=int)
args = parser.parse_args()

# Early exit if data hasn't been downloaded.
if not os.path.exists(os.path.join("data", "cifar10_train.tfrecord")):
    print("Training data does not exist! Run setup.py before running this script.")
    sys.exit(1)

# Early exit if training is complete.
last_checkpoint = latest_checkpoint("data")
if last_checkpoint and int(re.sub(".*ckpt-(\d+)", "\\1", last_checkpoint)) >= args.max_train_steps:
    print("Training is already complete. Run clean.py to start over, or increase --max_train_steps.")
    sys.exit(0)

cl = []
cl.append("python")
cl.append("src/train_squeezenet.py")
cl.append("--model_dir data")
cl.append("--train_tfrecord_filepaths data/cifar10_train.tfrecord")
cl.append("--validation_tfrecord_filepaths data/cifar10_test.tfrecord")
cl.append("--network squeezenet_cifar")
cl.append(f"--batch_size {args.batch_size}")
cl.append("--shuffle_buffer 1500")
cl.append(f"--data_format {args.data_format}")
cl.append("--target_image_size 32 32")
cl.append("--summary_interval 1")
cl.append(f"--max_train_steps {args.max_train_steps}")
subprocess.run(" ".join(cl), shell=True)