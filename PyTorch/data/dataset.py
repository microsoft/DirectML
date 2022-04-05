#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All rights reserved.

import argparse
import pathlib
import os
from torchvision import datasets
import wget
import zipfile

def get_current_dir():
    return str(pathlib.Path(__file__).parent.resolve())

def download_cifar_dataset():
    path = get_current_dir()
    datasets.CIFAR10(root=path, download=True)

def download_pennfudanped_dataset():
    path = get_current_dir()
    if (os.path.exists(os.path.join(path, 'PennFudanPed'))):
        print ("PennFundaPed dataset already downloaded and verified")
        return

    url='https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'
    print("Downloading PennFundaPed dataset\n")
    dataset_path = wget.download(url, out=path)
    try:
        with zipfile.ZipFile(os.path.join(path, dataset_path)) as z:
            z.extractall(path=path)
            print("\nExtracted PennFundaPed dataset")
    except:
        print("Invalid file")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dataset", help="datasets cifar10 or pennfudanped.", default="all")
    args = parser.parse_args()

    if args.dataset.lower() == 'all':
        download_cifar_dataset()
        download_pennfudanped_dataset()
    elif args.dataset.lower() == 'cifar10':
        download_cifar_dataset()
    elif args.dataset.lower() == 'pennfudanped':
        download_pennfudanped_dataset()
    else:
        raise Exception(f"Model {args.dataset} is not supported yet!")