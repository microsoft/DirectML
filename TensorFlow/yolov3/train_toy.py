import subprocess
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=1)
parser.add_argument("--epochs", default=4)
parser.add_argument("--mode", default="fit", choices=("fit", "eager_tf"))
parser.add_argument("--trace", action="store_true")
parser.add_argument("--lobe_data", action="store_true")
args = parser.parse_args()

cl_args = [
    "python train.py",
    f"--mode {args.mode}",
    "--transfer none",
    f"--batch_size {args.batch_size}",
    f"--epochs {args.epochs}",
    f"--trace={args.trace}"
]

# By default, use the single test image (data/mexico.jpg) as the dataset. The lobe_data
# option can be used to test with a slightly larger toy dataset in ../lobe folder/
if args.lobe_data:
    cl_args.append("--dataset", os.path.join("data", "toy_data.tfrecord"))
    cl_args.append("--classes", os.path.join("data", "toy_data.names"))
    cl_args.append("--num_classes 1")

    script_dir = os.path.dirname(os.path.realpath(__file__))
    toy_data_path = os.path.join(script_dir, "data", "toy_data.tfrecord")
    if not os.path.exists(toy_data_path):
        print("Converting toy data to TFRecord...")
        toy_data_script_path = os.path.join(script_dir, "tools", "create_toydata.py")
        subprocess.run(f"python {toy_data_script_path}", shell=True, check=True)

subprocess.run(" ".join(cl_args), shell=True, check=True)