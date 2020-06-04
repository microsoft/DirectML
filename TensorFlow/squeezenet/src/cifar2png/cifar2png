#!/usr/bin/env python
# coding: utf-8

import sys
import common.preprocess as preprocess


def main(argv=sys.argv[1:]):
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert CIFAR-10 and CIFAR-100 datasets into PNG images")
    parser.add_argument("dataset", type=str,
        choices=['cifar10', 'cifar100', 'cifar100superclass'],
        help="Specify dataset name. cifar10 or cifar100 or cifar100superclass")
    parser.add_argument("output", type=str,
        help="Path to save PNG converted dataset.")
    parser.add_argument("--name-with-batch-index", action="store_true",
        help="name image files based on batch name and index of cifar10/cifar100 dataset")
    args = parser.parse_args()

    preprocess.check_output_path(args.output)

    if preprocess.check_cifar(args.dataset):
        preprocess.save_cifar(args)


if __name__ == '__main__':
    main()
