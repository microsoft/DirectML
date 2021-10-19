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

from test_classification import main as classify_test


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--path", type=str, default="cifar-10-python", help="Path to cifar dataset.")
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='Batch size to train with.')
    parser.add_argument('--device', type=str, default='dml', help='The device to use for training.')
    parser.add_argument('--trace', type=bool, default=False, help='Trace performance.')
    args = parser.parse_args()

    batch_size = 1 if args.trace else args.batch_size

    classify_test(args.path, batch_size, args.device, 'squeezenet1_1', args.trace)

    
if __name__ == "__main__":
    main()