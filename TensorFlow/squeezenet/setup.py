#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All rights reserved.

import os
import subprocess

def find_cifar2png():
    for path in os.environ["PATH"].split(os.pathsep):
        for root,_,files in os.walk(path):
            for filename in files:
                if filename == "cifar2png":
                    return os.path.join(root, filename)
    return None

script_root = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_root)

script_path = os.path.join("src", "generate_cifar_tfrecords.py")
subprocess.run(f"python {script_path} --data_dir data", shell=True)

script_path = find_cifar2png()
if script_path is None:
    print("cifar2png not found! Please run pip install cifar2png")
    exit()

print("cifar2png found at ", script_path)

images_dir = os.path.join("data", "cifar10_images")
subprocess.run(f"python {script_path} cifar10 {images_dir}", shell=True)

if os.path.exists("cifar-10-python.tar.gz"):
    os.remove("cifar-10-python.tar.gz")