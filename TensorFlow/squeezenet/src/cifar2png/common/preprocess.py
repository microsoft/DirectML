#!/usr/bin/env python
# coding: utf-8

from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
logger.addHandler(handler)
logger.propagate = False

import hashlib
import math
import numpy as np
import os
import six
import sys
import tarfile
import requests

try:
    import cPickle as pickle
except:
    import pickle

try:
    from itertools import zip_longest
except:
    from itertools import izip_longest as zip_longest

from itertools import product
from collections import defaultdict
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from common.spinner import Spinner

CIFAR10_URL  = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR100_URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
CIFAR10_TAR_FILENAME  = 'cifar-10-python.tar.gz'
CIFAR100_TAR_FILENAME = 'cifar-100-python.tar.gz'
CIFAR10_TAR_MD5  = 'c58f30108f718f92721af3b95e74349a'
CIFAR100_TAR_MD5 = 'eb9058c3a382ffc7106e4002c42a8d85'

CIFAR10_TRAIN_DATA_NAMES = [
    'cifar-10-batches-py/data_batch_1',
    'cifar-10-batches-py/data_batch_2',
    'cifar-10-batches-py/data_batch_3',
    'cifar-10-batches-py/data_batch_4',
    'cifar-10-batches-py/data_batch_5'
]
CIFAR10_TEST_DATA_NAMES   = ['cifar-10-batches-py/test_batch']
CIFAR100_TRAIN_DATA_NAMES = ['cifar-100-python/train']
CIFAR100_TEST_DATA_NAMES  = ['cifar-100-python/test']

CIFAR10_LABELS_LIST = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
CIFAR100_LABELS_LIST = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]
CIFAR100_SUPERCLASS_LABELS_LIST = [
    'aquatic_mammals', 'fish', 'flowers', 'food_containers',
    'fruit_and_vegetables', 'household_electrical_devices',
    'household_furniture', 'insects', 'large_carnivores',
    'large_man-made_outdoor_things', 'large_natural_outdoor_scenes',
    'large_omnivores_and_herbivores', 'medium_mammals',
    'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals',
    'trees', 'vehicles_1', 'vehicles_2'
]
CIFAR100_CLASSES_LABELS_LIST = [
    ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    ['bottle', 'bowl', 'can', 'cup', 'plate'],
    ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    ['crab', 'lobster', 'snail', 'spider', 'worm'],
    ['baby', 'boy', 'girl', 'man', 'woman'],
    ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
]


def unpickle(dump):
    if six.PY2:
        data = pickle.loads(dump.read())
    elif six.PY3:
        data = pickle.loads(dump.read(), encoding='latin1')
    return data


def check_output_path(output):
    outputdir = Path(output)
    if outputdir.exists():
        logger.error("output dir `{}` already exists. Please specify a different output path".format(output))
        sys.exit(1)


# Reference: https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests/37573701
def download_with_progress(url, filename):
    logger.warning("Downloading {}".format(filename))
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0
    with open(filename, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size//block_size) , unit='KB', unit_scale=True):
            wrote = wrote  + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        logger.error("ERROR, something went wrong")
        sys.exit(1)


def download_cifar(dataset):
    if dataset == 'cifar10':
        download_with_progress(CIFAR10_URL, CIFAR10_TAR_FILENAME)
    elif dataset in ['cifar100', 'cifar100superclass']:
        download_with_progress(CIFAR100_URL, CIFAR100_TAR_FILENAME)


def check_cifar(dataset):
    if dataset == 'cifar10':
        cifar = Path(CIFAR10_TAR_FILENAME)
        md5sum = CIFAR10_TAR_MD5
    elif dataset in ['cifar100', 'cifar100superclass']:
        cifar = Path(CIFAR100_TAR_FILENAME)
        md5sum = CIFAR100_TAR_MD5

    if not cifar.is_file():
        logger.warning("{} does not exists.".format(cifar))
        download_cifar(dataset)

    cifar_md5sum = hashlib.md5(cifar.open('rb').read()).hexdigest()
    if md5sum != cifar_md5sum:
        logger.error("File `{0}` may be corrupted (wrong md5 checksum). Please delete `{0}` and retry".format(cifar))
        sys.exit(1)

    return True


def get_data_params(dataset):
    if dataset == 'cifar10':
        TARFILE = CIFAR10_TAR_FILENAME
        label_data = 'data'
        label_labels = 'labels'
        label_coarse = None
    elif dataset == 'cifar100':
        TARFILE = CIFAR100_TAR_FILENAME
        label_data = 'data'
        label_labels = 'fine_labels'
        label_coarse = None
    elif dataset == 'cifar100superclass':
        TARFILE = CIFAR100_TAR_FILENAME
        label_data = 'data'
        label_labels = 'fine_labels'
        label_coarse = 'coarse_labels'
    return TARFILE, label_data, label_labels, label_coarse


def get_datanames(dataset, mode):
    if dataset == 'cifar10':
        if mode == 'train':
            return CIFAR10_TRAIN_DATA_NAMES
        elif mode == 'test':
            return CIFAR10_TEST_DATA_NAMES
    elif dataset in ['cifar100', 'cifar100superclass']:
        if mode == 'train':
            return CIFAR100_TRAIN_DATA_NAMES
        elif mode == 'test':
            return CIFAR100_TEST_DATA_NAMES


def parse_cifar(dataset, mode):
    features = []
    labels = []
    coarse_labels = []
    batch_names = []

    TARFILE, label_data, label_labels, label_coarse = get_data_params(dataset)
    datanames = get_datanames(dataset, mode)

    try:
        spinner = Spinner(prefix="Loading {} data...".format(mode))
        spinner.start()
        tf = tarfile.open(TARFILE)
        for dataname in datanames:
            ti = tf.getmember(dataname)
            data = unpickle(tf.extractfile(ti))
            features.append(data[label_data])
            labels.append(data[label_labels])
            batch_names.extend([dataname.split('/')[1]] * len(data[label_data]))
            if dataset == 'cifar100superclass':
                coarse_labels.append(data[label_coarse])
        features = np.concatenate(features)
        features = features.reshape(features.shape[0], 3, 32, 32)
        features = features.transpose(0, 2, 3, 1).astype('uint8')
        labels = np.concatenate(labels)
        if dataset == 'cifar100superclass':
            coarse_labels = np.concatenate(coarse_labels)
        spinner.stop()
    except KeyboardInterrupt:
        spinner.stop()
        sys.exit(1)

    return features, labels, coarse_labels, batch_names


def save_cifar(args):
    dataset = args.dataset
    output = args.output
    if dataset == 'cifar10':
        LABELS = CIFAR10_LABELS_LIST
        LABELS_LIST = CIFAR10_LABELS_LIST
    elif dataset == 'cifar100':
        LABELS = CIFAR100_LABELS_LIST
        LABELS_LIST = CIFAR100_LABELS_LIST
    elif dataset == 'cifar100superclass':
        LABELS = []
        for i in zip(CIFAR100_SUPERCLASS_LABELS_LIST, CIFAR100_CLASSES_LABELS_LIST):
            for j in product([i[0]], i[1]):
                LABELS.append('/'.join(j))
        LABELS_LIST = CIFAR100_LABELS_LIST
        COARSE_LABELS_LIST = CIFAR100_SUPERCLASS_LABELS_LIST

    for mode in ['train', 'test']:
        for label in LABELS:
            dirpath = os.path.join(output, mode, label)
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)

        features, labels , coarse_labels, batch_names = parse_cifar(dataset, mode)

        label_count = defaultdict(int)
        batch_count = defaultdict(int)
        for feature, label, coarse_label, batch_name in tqdm(zip_longest(features, labels, coarse_labels, batch_names), total=len(labels), desc="Saving {} images".format(mode)):
            label_count[label] += 1
            if args.name_with_batch_index:
                if args.dataset == 'cifar10':
                    filename = '%s_index_%04d.png' % (batch_name, batch_count[batch_name])
                else:
                    filename = '%s_index_%05d.png' % (batch_name, batch_count[batch_name])
            else:
                filename = '%04d.png' % label_count[label]
            batch_count[batch_name] += 1

            if dataset == 'cifar100superclass':
                filepath = os.path.join(output, mode, COARSE_LABELS_LIST[coarse_label], LABELS_LIST[label], filename)
            else:
                filepath = os.path.join(output, mode, LABELS_LIST[label], filename)
            image = Image.fromarray(feature)
            image = image.convert('RGB')
            image.save(filepath)
