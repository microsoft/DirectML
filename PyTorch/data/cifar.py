#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All rights reserved.

import argparse
import pathlib
import os
import subprocess
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose, transforms

def get_training_path(args):
    if (os.path.isabs(args.path)):
        return args.path
    else:
        return str(os.path.join(pathlib.Path(__file__).parent.resolve(), args.path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("-path", help="Path to cifar dataset.", default="cifar-10-python")
    args = parser.parse_args()

    path = get_training_path(args)
    datasets.CIFAR10(root=path, download=True)