cifar2png
=========

Convert CIFAR-10 or CIFAR-100 dataset into PNG images.


# Install

```bash
$ pip install cifar2png
```


# Usage

`$ cifar2png <dataset> <output_dir> [--name-with-batch-index]`

- `dataset`: Specify `cifar10` or `cifar100` or `cifar100superclass`
- `output_dir`: Path to save PNG converted dataset (The directory will be created automatically).
- `--name-with-batch-index`: (optional) Name image files based on batch name and index of cifar10/cifar100 dataset.

Automatically download `cifar-10-python.tar.gz` or `cifar-100-python.tar.gz` to the current directory from [CIFAR-10 and CIFAR-100 datasets] when you run this tool.


# Examples


## CIFAR-10

`$ cifar2png cifar10 path/to/cifar10png`


## CIFAR-10 with naming option

`$ cifar2png cifar10 path/to/cifar10png --name-with-batch-index`


## CIFAR-100

`$ cifar2png cifar100 path/to/cifar100png`


## CIFAR-100 with superclass

`$ cifar2png cifar100superclass path/to/cifar100png`


# Structure of output directory

## CIFAR-10 and CIFAR-100

PNG images of CIFAR-10 are saved in 10 subdirectories of each label under the `test` and `train` directories as below.  
(CIFAR-100 are saved in the same way with 100 subdirectories)

```bash
$ tree -d path/to/cifar10png
path/to/cifar10png
├── test
│   ├── airplane
│   ├── automobile
│   ├── bird
│   ├── cat
│   ├── deer
│   ├── dog
│   ├── frog
│   ├── horse
│   ├── ship
│   └── truck
└── train
    ├── airplane
    ├── automobile
    ├── bird
    ├── cat
    ├── deer
    ├── dog
    ├── frog
    ├── horse
    ├── ship
    └── truck
```

```bash
$ tree path/to/cifar10png/test/airplane
path/to/cifar10png/test/airplane
├── 0001.png
├── 0002.png
├── 0003.png
(..snip..)
├── 0998.png
├── 0999.png
└── 1000.png
```

When dataset created using the `--name-with-batch-index` option.

```bash
$ tree path/to/cifar10png/train/airplane
path/to/cifar10png/train/airplane
├── data_batch_1_index_0029.png
├── data_batch_1_index_0030.png
├── data_batch_1_index_0035.png
(..snip..)
├── data_batch_5_index_9941.png
├── data_batch_5_index_9992.png
└── data_batch_5_index_9994.png
```

## CIFAR-100 with superclass

PNG images of CIFAR-100 with superclass are saved in each label directories under the superclass subdirectories under the test and train directories as below.

```
$ tree -d path/to/cifar100png
path/to/cifar100png
├── test
│   ├── aquatic_mammals
│   │   ├── beaver
│   │   ├── dolphin
│   │   ├── otter
│   │   ├── seal
│   │   └── whale
│   ├── fish
│   │   ├── aquarium_fish
│   │   ├── flatfish
│   │   ├── ray
│   │   ├── shark
│   │   └── trout
│   ├── flowers
│   │   ├── orchid
│   │   ├── poppy
│   │   ├── rose
│   │   ├── sunflower
│   │   └── tulip
(..snip..)
    ├── trees
    │   ├── maple_tree
    │   ├── oak_tree
    │   ├── palm_tree
    │   ├── pine_tree
    │   └── willow_tree
    ├── vehicles_1
    │   ├── bicycle
    │   ├── bus
    │   ├── motorcycle
    │   ├── pickup_truck
    │   └── train
    └── vehicles_2
        ├── lawn_mower
        ├── rocket
        ├── streetcar
        ├── tank
        └── tractor
```


[CIFAR-10 and CIFAR-100 datasets]: https://www.cs.toronto.edu/~kriz/cifar.html
