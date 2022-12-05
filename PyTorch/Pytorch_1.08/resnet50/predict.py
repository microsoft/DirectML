#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All rights reserved.

import argparse
import subprocess
import os
import pathlib

import sys
classification_folder = str(os.path.join(pathlib.Path(__file__).parent.parent.resolve(), 'classification'))
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, classification_folder)

from test_classification import predict

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--image", type=str, help="Image to classify.")
    parser.add_argument('--device', type=str, default='dml', help='The device to use for training.')
    args = parser.parse_args()

    predict(args.image, 'resnet50', args.device)


if __name__ == "__main__":
    main()