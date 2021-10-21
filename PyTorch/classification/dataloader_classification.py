import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose, transforms
import torchvision.models as models
import collections
import matplotlib.pyplot as plt
import argparse
import time
import os
import pathlib

def get_pytorch_root(path):
    return pathlib.Path(__file__).parent.parent.resolve()


def get_pytorch_data():
    return str(os.path.join(pathlib.Path(__file__).parent.parent.resolve(), 'data'))


def get_data_path(path):
    if (os.path.isabs(path)):
        return path
    else:
        return str(os.path.join(get_pytorch_data(), path))


def print_dataloader(dataloader, mode):
    for X, y in dataloader:
        print("\t{} data X [N, C, H, W]: \n\t\tshape={}, \n\t\tdtype={}".format(mode, X.shape, X.dtype))
        print("\t{} data Y: \n\t\tshape={}, \n\t\tdtype={}".format(mode, y.shape, y.dtype))
        break


def create_training_data_transform():
    return transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def create_training_dataloader(path, batch_size):
    path = get_data_path(path)
    print('Loading the training dataset from: {}'.format(path))
    train_transform = create_training_data_transform()       
    training_set = datasets.CIFAR10(root=path, train=True, download=False, transform=train_transform)
    data_loader = DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True, num_workers=0)
    print_dataloader(data_loader, 'Train')
    return data_loader


def create_testing_data_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def create_testing_dataloader(path, batch_size):
    path = get_data_path(path)
    print('Loading the testing dataset from: {}'.format(path))
    test_transform = create_testing_data_transform()
    test_set = datasets.CIFAR10(root=path, train=False, download=False, transform=test_transform)
    data_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    print_dataloader(data_loader, 'Test')
    return data_loader