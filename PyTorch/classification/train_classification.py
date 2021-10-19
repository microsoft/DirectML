#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All rights reserved.

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
import test_classification
import dataloader_classification
import torch.autograd.profiler as profiler


def train(dataloader, model, device, loss, learning_rate, momentum, weight_decay, trace):
    size = len(dataloader.dataset)

    # Define optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay)
        
    optimize_after_batches = 1

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        if (trace):
            with profiler.profile(record_shapes=True, with_stack=True, profile_memory=True) as prof:
                with profiler.record_function("model_inference"):
                    # Compute loss and perform backpropagation
                    batch_loss = loss(model(X), y)
                    batch_loss.backward()

                    if batch % optimize_after_batches == 0:
                        optimizer.step()
                        optimizer.zero_grad()
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1000))
            break;
        else:
            # Compute loss and perform backpropagation
            batch_loss = loss(model(X), y)
            batch_loss.backward()

            if batch % optimize_after_batches == 0:
                optimizer.step()
                optimizer.zero_grad()

        if batch % 100 == 0:
            batch_loss_cpu, current = batch_loss.to('cpu'), batch * len(X)
            print(f"loss: {batch_loss_cpu.item():>7f}  [{current:>5d}/{size:>5d}]")


def main(path, batch_size, epochs, learning_rate,
         momentum, weight_decay, device, model_str, save_model, trace):
    batch_size = 1 if trace else batch_size
    epochs = 1 if trace else epochs

    # Load the dataset
    training_dataloader = dataloader_classification.create_training_dataloader(path, batch_size)
    testing_dataloader = dataloader_classification.create_testing_dataloader(path, batch_size)

    # Create the device
    device = torch.device(device)

    # Load the model on the device
    start = time.time()

    if (model_str == 'squeezenet1_1'):
        model = models.squeezenet1_1(num_classes=10).to(device)
    elif (model_str == 'resnet50'):
        model = models.resnet50(num_classes=10).to(device)
    else:
        raise Exception(f"Model {model_str} is not supported yet!")

    print('Finished moving {} to device: {} in {}s.'.format(model_str, device, time.time() - start))
 
    cross_entropy_loss = nn.CrossEntropyLoss().to(device)

    highest_accuracy = 0
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        # Train
        train(training_dataloader,
              model,
              device,
              cross_entropy_loss,
              learning_rate,
              momentum,
              weight_decay,
              trace)

        if not trace:
            # Test
            highest_accuracy = test_classification.eval(testing_dataloader,
                                    model_str,
                                    model,
                                    device,
                                    cross_entropy_loss,
                                    highest_accuracy,
                                    save_model,
                                    False)

    print("Done! with highest_accuracy: ", highest_accuracy)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--path", type=str, default="cifar-10-python", help="Path to cifar dataset.")
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='Batch size to train with.')
    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='The number of epochs to train for.')
    parser.add_argument('--learning_rate', type=float, default=0.001, metavar='LR', help='The learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='The percentage of past parameters to store.')
    parser.add_argument('--weight_decay', default=0.0001, type=float, help='The parameter to decay weights.')
    parser.add_argument('--device', type=str, default='dml', help='The device to use for training.')
    parser.add_argument('--model', type=str, default='squeezenet1_1', help='The model to use.')
    parser.add_argument('--save_model', action='store_true', help='save model state_dict to file')
    parser.add_argument('--trace', type=bool, default=False, help='Trace performance.')
    args = parser.parse_args()

    main(args.path, args.batch_size, args.epochs, args.learning_rate,
         args.momentum, args.weight_decay, args.device, args.model, args.save_model, args.trace)