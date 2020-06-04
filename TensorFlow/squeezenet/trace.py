#!/usr/bin/env python
import subprocess
import argparse
import shutil
import glob
import os
import datetime
import sys

script_root = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_root)

# Early exit if data hasn't been downloaded.
if not os.path.exists(os.path.join("data", "cifar10_train.tfrecord")):
    print("Training data does not exist! Run setup.py before running this script.")
    sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32)
parser.add_argument("--data_format", choices=("NCHW", "NHWC"), default="NCHW")
parser.add_argument("--max_train_steps", default=10)
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--name", "-n")
args = parser.parse_args()

subprocess.run("python clean.py", shell=True)

cl = []
cl.append("python")
cl.append("src/train_squeezenet.py")
cl.append("--model_dir data")
cl.append("--train_tfrecord_filepaths data/cifar10_train.tfrecord")
cl.append("--validation_tfrecord_filepaths data/cifar10_test.tfrecord")
cl.append("--network squeezenet_cifar")
cl.append("--shuffle_buffer 1500")
cl.append("--target_image_size 32 32")
cl.append("--summary_interval 1")
cl.append("--validation_interval 0")
cl.append("--keep_last_n_checkpoints 0")
cl.append("--trace")
cl.append(f"--data_format {args.data_format}")
cl.append(f"--batch_size {args.batch_size}")
cl.append(f"--max_train_steps {args.max_train_steps}")
if (args.cpu):
    cl.append("--clone_on_cpu")
subprocess.run(" ".join(cl), shell=True, check=True)

traces = glob.glob("data/cifar_trace*")
if traces:
    timestr = datetime.datetime.now().strftime("%y%m%d-%H%M%S")

    if not args.name:
        if "CONDA_DEFAULT_ENV" in os.environ:
            args.name = os.environ["CONDA_DEFAULT_ENV"] + "_"
    else:
        args.name = args.name + "_"

    output_dir = os.path.join("traces", f"cifar_{args.name}{args.batch_size}_{args.data_format}_{timestr}")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for i in range(0,len(traces)):
        shutil.copy(traces[i], output_dir)

    timing_script = os.path.join("..", "helpers", "trace_times.py")
    subprocess.run(f"python {timing_script} --traces_dir {output_dir}", shell=True, check=True)