import subprocess
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8)
parser.add_argument("--epochs", default=10)
parser.add_argument("--mode", default="fit", choices=("fit", "eager_tf"))
parser.add_argument("--trace", action="store_true")
args = parser.parse_args()

cl = " ".join([
    "python train.py",
    "--dataset", os.path.join("data", "voc2012_train.tfrecord"),
    "--val_dataset", os.path.join("data", "voc2012_val.tfrecord"),
    "--classes", os.path.join("data", "voc2012.names"),
    "--num_classes 20",
    f"--mode {args.mode}",
    "--transfer none",
    f"--batch_size {args.batch_size}",
    f"--epochs {args.epochs}",
    f"--trace={args.trace}"
])
subprocess.run(cl, shell=True, check=True)