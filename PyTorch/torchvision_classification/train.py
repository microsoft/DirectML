import argparse
import subprocess
import os
import pathlib

import sys
classification_folder = str(os.path.join(pathlib.Path(__file__).parent.parent.resolve(), 'classification'))
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, classification_folder)

from train_classification import main as classify_train

classification_models = ['resnet18',
                        'alexnet',
                        'vgg16',
                        'squeezenet1_0',
                        'densenet161',
                        'inception_v3',
                        'googlenet',
                        'shufflenet_v2_x1_0',
                        'mobilenet_v2',
                        'mobilenet_v3_large',
                        'mobilenet_v3_small',
                        'resnext50_32x4d',
                        'wide_resnet50_2',
                        'mnasnet1_0']

def main():
    
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--path", type=str, default="cifar-10-python", help="Path to cifar dataset.")
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='Batch size to train with.')
    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='The number of epochs to train for.')
    parser.add_argument('--learning_rate', type=float, default=0.001, metavar='LR', help='The learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='The percentage of past parameters to store.')
    parser.add_argument('--weight_decay', default=0.0001, type=float, help='The parameter to decay weights.')
    parser.add_argument('--device', type=str, default='dml', help='The device to use for training.')
    parser.add_argument('--model', type=str, default='squeezenet1_0', help='The model to use.')
    parser.add_argument('--save_model', action='store_true', help='Save the model state_dict to file')
    parser.add_argument('--trace', type=bool, default=False, help='Trace performance.')
    args = parser.parse_args()
    print (args)

    classify_train(args.path, args.batch_size, args.epochs, args.learning_rate,
            args.momentum, args.weight_decay, args.device, args.model, args.save_model, args.trace)

    

if __name__ == "__main__":
    main()