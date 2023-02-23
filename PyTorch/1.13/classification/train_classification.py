#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All rights reserved.

import torch
import torch_directml
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
from test_classification import get_model
import torch.autograd.profiler as profiler

def select_device(device=''):
    if device.lower() == 'cuda':
        if not torch.cuda.is_available():
            print ("torch.cuda not available")
            return torch.device('cpu')    
        else:
            return torch.device('cuda:0')
    if device.lower() == 'dml':
        return torch_directml.device(torch_directml.default_device())
    else:
        return torch.device('cpu')

def train(dataloader, model, device, loss, learning_rate, momentum, weight_decay, trace, model_str, ci_train):
    size = len(dataloader.dataset)

    # Define optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay)
        
    optimize_after_batches = 1
    start = time.time()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)

        if (trace):
            with profiler.profile(record_shapes=True, with_stack=True, profile_memory=True) as prof:
                with profiler.record_function("model_inference"):
                    # Compute loss and perform backpropagation
                    if (model_str == 'inception_v3'):
                        pred, _ = model(X)
                    elif (model_str == 'googlenet'):
                        pred, _, _ = model(X)
                    else:
                        pred = model(X)

                    batch_loss = loss(pred, y)
                    batch_loss.backward()

                    if batch % optimize_after_batches == 0:
                        optimizer.step()
                        optimizer.zero_grad()
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1000))
            break
        else:
            # Compute loss and perform backpropagation
            if (model_str == 'inception_v3'):
                outputs, aux_outputs = model(X)
                loss1 = loss(outputs, y)
                loss2 = loss(aux_outputs, y)
                batch_loss = loss1 + 0.4*loss2
            elif (model_str == 'googlenet'):
                outputs, aux_outputs_1, aux_outputs_2 = model(X)
                loss1 = loss(outputs, y)
                loss2 = loss(aux_outputs_1, y)
                loss3 = loss(aux_outputs_2, y)
                batch_loss = loss1 + 0.3*loss2 + 0.3*loss3
            else:
                pred = model(X)
                batch_loss = loss(model(X), y)
            batch_loss.backward()

            if batch % optimize_after_batches == 0:
                optimizer.step()
                optimizer.zero_grad()
            
        if (batch+1) % 100 == 0:
            batch_loss_cpu, current = batch_loss.to('cpu'), (batch+1) * len(X)
            print(f"loss: {batch_loss_cpu.item():>7f}  [{current:>5d}/{size:>5d}] in {time.time() - start:>5f}s")
            start = time.time()

        if ci_train:
            print(f"train [{len(X):>5d}/{size:>5d}] in {time.time() - start:>5f}s")
            break


def main(path, batch_size, epochs, learning_rate,
         momentum, weight_decay, device, model_str, save_model, trace, ci_train=False):
    if trace:
        if model_str == 'inception_v3':
            batch_size = 3
        else:
            batch_size = 1
    epochs = 1 if trace or ci_train else epochs
    
    input_size = 299 if model_str == 'inception_v3' else 224

    model = get_model(model_str, device)

    # Load the dataset
    training_dataloader = dataloader_classification.create_training_dataloader(path, batch_size, input_size)
    testing_dataloader = dataloader_classification.create_testing_dataloader(path, batch_size, input_size)

    # Load the model on the device
    start = time.time()
    
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
              trace,
              model_str,
              ci_train)

        if not trace and not ci_train:
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
    parser.add_argument('--model', type=str, default='', help='The model to use.')
    parser.add_argument('--save_model', action='store_true', help='save model state_dict to file')
    parser.add_argument('--trace', type=bool, default=False, help='Trace performance.')
    args = parser.parse_args()

    print (args)
    device = select_device(args.device)
    main(args.path, args.batch_size, args.epochs, args.learning_rate,
         args.momentum, args.weight_decay, device, args.model, args.save_model, args.trace)