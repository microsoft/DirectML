import subprocess
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=4)
parser.add_argument("--epochs", default=10)
args = parser.parse_args()

cl = " ".join([
    "python train.py",
    "--dataset", os.path.join("data", "voc2012_train.tfrecord"),
    "--val_dataset", os.path.join("data", "voc2012_val.tfrecord"),
    "--num_samples 5717",
    "--num_val_samples 5823",
    "--classes", os.path.join("data", "voc2012.names"),
    "--num_classes 20",
    "--mode eager_tf",
    "--transfer none",
    f"--batch_size {args.batch_size}",
    f"--epochs {args.epochs}"
])
subprocess.run(cl, shell=True, check=True)