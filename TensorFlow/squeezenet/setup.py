#!/usr/bin/env python
import os
import subprocess

script_root = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_root)

script_path = os.path.join("src", "generate_cifar_tfrecords.py")
subprocess.run(f"python {script_path} --data_dir data", shell=True)

script_path = os.path.join("src", "cifar2png", "cifar2png")
images_dir = os.path.join("data", "cifar10_images")
subprocess.run(f"python {script_path} cifar10 {images_dir}", shell=True)

if os.path.exists("cifar-10-python.tar.gz"):
    os.remove("cifar-10-python.tar.gz")