import torch
from torch import nn
from torch.utils import data
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
import dataloader_classification
import torch.autograd.profiler as profiler
from PIL import Image
from os.path import exists

def get_checkpoint_folder(model_str, device):
    checkpoint_folder = str(os.path.join(pathlib.Path(__file__).parent.parent.resolve(),
                    'checkpoints', model_str, str(device)))
    os.makedirs(checkpoint_folder, exist_ok=True)
    return str(os.path.join(checkpoint_folder, 'checkpoint.pth'))

def eval(dataloader, model_str, model, device, loss, highest_accuracy, save_model, trace):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    # Switch model to evaluation mode
    model.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            
            # Evaluate the model on the test input
            if (trace):
                with profiler.profile(record_shapes=True, with_stack=True, profile_memory=True) as prof:
                    with profiler.record_function("model_inference"):
                        pred = model(X)
                print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1000))
                break
            else:
                pred = model(X)

            test_loss += loss(pred, y).to("cpu")
            correct += (pred.to("cpu").argmax(1) == y.to("cpu")).type(torch.float).sum()

    if not trace:
        test_loss /= num_batches
        correct /= size
        
        if (correct.item() > highest_accuracy):
            highest_accuracy = correct.item()
            print("current highest_accuracy: ", highest_accuracy)
            
            # save model
            if save_model:
                state_dict = collections.OrderedDict()
                for key in model.state_dict().keys():
                    state_dict[key] = model.state_dict()[key].to("cpu")
                checkpoint = get_checkpoint_folder(model_str, device)
                torch.save(state_dict, checkpoint)

        print(f"Test Error: \n Accuracy: {(100*correct.item()):>0.1f}%, Avg loss: {test_loss.item():>8f} \n")

    return highest_accuracy


def get_model(model_str, device):
    if (model_str == 'squeezenet1_1'):
        model = models.squeezenet1_1(num_classes=10).to(device)
    elif (model_str == 'resnet50'):
        model = models.resnet50(num_classes=10).to(device)
    else:
        raise Exception(f"Model {model_str} is not supported yet!")
    
    checkpoint = get_checkpoint_folder(model_str, device)
    if (exists(checkpoint)):
        model.load_state_dict(torch.load(checkpoint))

    return model

def preprocess(filename, device):
    input_image = Image.open(filename)
    preprocess_transform = dataloader_classification.create_testing_data_transform()
    input_tensor = preprocess_transform(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    input_batch = input_batch.to(device)
    return input_batch

def predict(filename, model_str, device):
    # Get the model
    model = get_model(model_str, device)
    model.eval()

    # Preprocess input
    input = preprocess(filename, device)

    # Evaluate
    with torch.no_grad():
        pred = model(input).to('cpu')

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(pred[0], dim=0)

    data_folder = dataloader_classification.get_pytorch_data()
    classes_file = str(os.path.join(data_folder, 'imagenet_classes.txt'))
    with open(classes_file, "r") as f:
        categories = [s.strip() for s in f.readlines()]
        # Show top categories per image
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        for i in range(top5_prob.size(0)):
            print(categories[top5_catid[i]], top5_prob[i].item())


def main(path, batch_size, device, model_str, trace):
    # Load the dataset
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

    # Test
    highest_accuracy = eval(testing_dataloader,
                            model_str,
                            model,
                            device,
                            cross_entropy_loss,
                            0,
                            False,
                            trace)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--path", type=str, default="cifar-10-python", help="Path to cifar dataset.")
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='Batch size to train with.')
    parser.add_argument('--device', type=str, default='dml', help='The device to use for training.')
    parser.add_argument('--model', type=str, default='squeezenet1_1', help='The model to use.')
    args = parser.parse_args()

    main(args.path, args.batch_size, args.device, args.model)