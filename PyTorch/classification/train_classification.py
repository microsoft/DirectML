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

def select_device(device=''):
    if device == 'cuda':
        if not torch.cuda.is_available():
            print ("torch.cuda not available")
            return torch.device('cpu')    
        else:
            return torch.device('cuda:0')
    if device.lower() == 'dml':
        return torch.device('dml')
    else:
        return torch.device('cpu')

def train(dataloader, model, device, loss, learning_rate, momentum, weight_decay, trace, model_str):
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
            prof.export_chrome_trace(trace_des_path)
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
            batch_loss_cpu, current = batch_loss.to('cpu'), batch * len(X)
            print(f"loss: {batch_loss_cpu.item():>7f}  [{current:>5d}/{size:>5d}] in {time.time() - start:>5f}s")
            start = time.time()


def main(path, batch_size, epochs, learning_rate,
         momentum, weight_decay, device, model_str, save_model, trace):
    if trace:
        if model_str == 'inception_v3':
            batch_size = 3
        else:
            batch_size = 1
        ephochs = 1
    
    input_size = 299 if model_str == 'inception_v3' else 224

    if (model_str == 'squeezenet1_1'):
        model = models.squeezenet1_1(num_classes=10).to(device)
    elif (model_str == 'resnet50'):
        model = models.resnet50(num_classes=10).to(device)
    elif (model_str == 'squeezenet1_0'):
        model = models.squeezenet1_0(num_classes=10).to(device)
    elif (model_str == 'resnet18'):
        model = models.resnet18(num_classes=10).to(device)
    elif (model_str == 'alexnet'):
        model = models.alexnet(num_classes=10).to(device)
    elif (model_str == 'vgg16'):
        model = models.vgg16(num_classes=10).to(device)
    elif (model_str == 'densenet161'):
        model = models.densenet161(num_classes=10).to(device)
    elif (model_str == 'inception_v3'):
        model = models.inception_v3(num_classes=10).to(device)
    elif (model_str == 'googlenet'):
        model = models.googlenet(num_classes=10).to(device)
    elif (model_str == 'shufflenet_v2_x1_0'):
        model = models.shufflenet_v2_x1_0(num_classes=10).to(device)
    elif (model_str == 'mobilenet_v2'):
        model = models.mobilenet_v2(num_classes=10).to(device)
    elif (model_str == 'mobilenet_v3_large'):
        model = models.mobilenet_v3_large(num_classes=10).to(device)
    elif (model_str == 'mobilenet_v3_small'):
        model = models.mobilenet_v3_small(num_classes=10).to(device)
    elif (model_str == 'resnext50_32x4d'):
        model = models.resnext50_32x4d(num_classes=10).to(device)
    elif (model_str == 'wide_resnet50_2'):
        model = models.wide_resnet50_2(num_classes=10).to(device)
    elif (model_str == 'mnasnet1_0'):
        model = models.mnasnet1_0(num_classes=10).to(device)
    else:
        raise Exception(f"Model {model_str} is not supported yet!")

    # Load the dataset
    training_dataloader = dataloader_classification.create_training_dataloader(path, batch_size, input_size)
    testing_dataloader = dataloader_classification.create_testing_dataloader(path, batch_size, input_size)

    # Create the device
    device = select_device(device)

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
              model_str)

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
    parser.add_argument('--model', type=str, default='', help='The model to use.')
    parser.add_argument('--save_model', action='store_true', help='save model state_dict to file')
    parser.add_argument('--trace', type=bool, default=False, help='Trace performance.')
    args = parser.parse_args()

    if (not model_str):
        print("please specify the model for model list: ")
    main(args.path, args.batch_size, args.epochs, args.learning_rate,
         args.momentum, args.weight_decay, args.device, args.model, args.save_model, args.trace)