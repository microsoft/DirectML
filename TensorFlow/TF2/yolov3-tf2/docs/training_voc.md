# Training Instruction

## VOC 2012 Dataset from Scratch

Full instruction on how to train using VOC 2012 from scratch

Requirement:
  1. Able to detect image using pretrained darknet model
  2. Many Gigabytes of Disk Space
  3. High Speed Internet Connection Preferred
  4. GPU Preferred


### 1. Download Dataset

You can read the full description of dataset [here](http://host.robots.ox.ac.uk/pascal/VOC/)
```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -O ./data/voc2012_raw.tar
mkdir -p ./data/voc2012_raw
tar -xf ./data/voc2012_raw.tar -C ./data/voc2012_raw
ls ./data/voc2012_raw/VOCdevkit/VOC2012 # Explore the dataset
```

### 2. Transform Dataset

See tools/voc2012.py for implementation, this format is based on [tensorflow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). Many fields 
are not required, I left them there for compatibility with official API.

```bash
python tools/voc2012.py \
  --data_dir './data/voc2012_raw/VOCdevkit/VOC2012' \
  --split train \
  --output_file ./data/voc2012_train.tfrecord

python tools/voc2012.py \
  --data_dir './data/voc2012_raw/VOCdevkit/VOC2012' \
  --split val \
  --output_file ./data/voc2012_val.tfrecord
```

You can visualize the dataset using this tool
```
python tools/visualize_dataset.py --classes=./data/voc2012.names
```

It will output one random image with label to `output.jpg`

### 3. Training

You can adjust the parameters based on your setup

#### With Transfer Learning

This step requires loading the pretrained darknet (feature extractor) weights.
```
wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights
python convert.py
python detect.py --image ./data/meme.jpg # Sanity check

python train.py \
	--dataset ./data/voc2012_train.tfrecord \
	--val_dataset ./data/voc2012_val.tfrecord \
	--classes ./data/voc2012.names \
	--num_classes 20 \
	--mode fit --transfer darknet \
	--batch_size 16 \
	--epochs 10 \
	--weights ./checkpoints/yolov3.tf \
	--weights_num_classes 80 
```

Original pretrained yolov3 has 80 classes, here we demonstrated how to
do transfer learning on 20 classes.

#### Training from random weights (NOT RECOMMENDED)
Training from scratch is very difficult to converge
The original paper trained darknet 
on imagenet before training the whole network as well.

```bash
python train.py \
	--dataset ./data/voc2012_train.tfrecord \
	--val_dataset ./data/voc2012_val.tfrecord \
	--classes ./data/voc2012.names \
	--num_classes 20 \
	--mode fit --transfer none \
	--batch_size 16 \
	--epochs 10 \
```

I have tested this works 100% with correct loss and converging over time.
Each epoch takes around 10 minutes on single AWS p2.xlarge (Nvidia K80 GPU) Instance.

You might see warnings or error messages during training, they are not critical dont' worry too much about them.
There might be a long wait time between each epoch becaues we are calculating validation loss.

### 4. Inference

```bash
# detect from images
python detect.py \
	--classes ./data/voc2012.names \
	--num_classes 20 \
	--weights ./checkpoints/yolov3_train_5.tf \
	--image ./data/street.jpg

# detect from validation set
python detect.py \
	--classes ./data/voc2012.names \
	--num_classes 20 \
	--weights ./checkpoints/yolov3_train_5.tf \
	--tfrecord ./data/voc2012_val.tfrecord
```

You should see some detect objects in the standard output and the visualization at `output.jpg`.
this is just a proof of concept, so it won't be as good as pretrained models.
In my experience, you might need lower score score thershold if you didn't train it enough.

