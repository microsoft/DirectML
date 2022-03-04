# import torchvision
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# # load a model pre-trained pre-trained on COCO
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# # replace the classifier with a new one, that has
# # num_classes which is user-defined
# num_classes = 2  # 1 class (person) + background
# # get number of input features for the classifier
# in_features = model.roi_heads.box_predictor.cls_score.in_features
# # replace the pre-trained head with a new one
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

# import torchvision
# from torchvision.models.detection import FasterRCNN
# from torchvision.models.detection.rpn import AnchorGenerator

# # load a pre-trained model for classification and return
# # only the features
# backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# # FasterRCNN needs to know the number of
# # output channels in a backbone. For mobilenet_v2, it's 1280
# # so we need to add it here
# backbone.out_channels = 1280

# # let's make the RPN generate 5 x 3 anchors per spatial
# # location, with 5 different sizes and 3 different aspect
# # ratios. We have a Tuple[Tuple[int]] because each feature
# # map could potentially have different sizes and
# # aspect ratios 
# anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
#                                    aspect_ratios=((0.5, 1.0, 2.0),))

# # let's define what are the feature maps that we will
# # use to perform the region of interest cropping, as well as
# # the size of the crop after rescaling.
# # if your backbone returns a Tensor, featmap_names is expected to
# # be [0]. More generally, the backbone should return an
# # OrderedDict[Tensor], and in featmap_names you can choose which
# # feature maps to use.
# roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
#                                                 output_size=7,
#                                                 sampling_ratio=2)

# # put the pieces together inside a FasterRCNN model
# model = FasterRCNN(backbone,
#                    num_classes=2,
#                    rpn_anchor_generator=anchor_generator,
#                    box_roi_pool=roi_pooler)
import torch.autograd.profiler as profiler
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PennFudanDataset import *

# import os
# input(f"pid: {os.getpid()}")

def get_instance_segmentation_model(num_classes, model_str='maskrcnn'):
    if model_str == 'maskrcnn':
      # load an instance segmentation model pre-trained on COCO
      model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

      # get the number of input features for the classifier
      in_features = model.roi_heads.box_predictor.cls_score.in_features
      # replace the pre-trained head with a new one
      model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

      # now get the number of input features for the mask classifier
      in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
      hidden_layer = 256
      # and replace the mask predictor with a new one
      model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        num_classes)
    if model_str == 'fastrcnn':
      # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
      model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)

      # replace the classifier with a new one, that has
      # num_classes which is user-defined
      num_classes = 2  # 1 class (person) + background
      # get number of input features for the classifier
      in_features = model.roi_heads.box_predictor.cls_score.in_features
      # replace the pre-trained head with a new one
      model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
    return model
  
from engine import train_one_epoch, evaluate
import utils
import transforms as T


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# use our dataset and defined transformations
dataset = PennFudanDataset('E:\\xianz\\DirectML\\PyTorch\\objectDetection\\PennFudanPed', get_transform(train=True))
dataset_test = PennFudanDataset('E:\\xianz\\DirectML\\PyTorch\\objectDetection\\PennFudanPed', get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

device = torch.device('dml') # if torch.cuda.is_available() else torch.device('cpu')
print (device)

# our dataset has two classes only - background and person
num_classes = 2

# get the model using our helper function
# model = get_instance_segmentation_model(num_classes, 'maskrcnn')

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                                step_size=3,
#                                                gamma=0.1)

# # let's train it for 10 epochs
# from torch.optim.lr_scheduler import StepLR
num_epochs = 1

if __name__ == '__main__':
  import math
  import sys
  for epoch in range(num_epochs):
      model.train()

      for batch, (images, targets) in enumerate(data_loader):
        #   images = list(image.to(device) for image in images)
        #   targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #   for t in targets:
        #     for k,v in t.items():
        #       print (k)
        #       print (v.to("cpu"))


          images, boxes = torch.rand(1, 3, 600, 1200).to(device), torch.rand(1, 11, 4).sort().values.to(device)
          labels = torch.randint(1, 91, (1, 11)).to(device)
          images = list(image for image in images)
          targets = []

          for i in range(len(images)):
            d = {}
            d['boxes'] = boxes[i]
            d['labels'] = labels[i]
            targets.append(d)

          with profiler.profile(record_shapes=True, with_stack=True, profile_memory=True) as prof:
              with profiler.record_function("model_inference"):
                loss_dict = model(images, targets)

                losses = sum(loss for loss in loss_dict.values())

                # # reduce losses over all GPUs for logging purposes
                # loss_dict_reduced = utils.reduce_dict(loss_dict)
                # losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                # loss_value = losses_reduced.item()

                # if not math.isfinite(loss_value):
                #     print("Loss is {}, stopping training".format(loss_value))
                #     print(loss_dict_reduced)
                #     sys.exit(1)

                print ("start backward!!")
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
          print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1000))
          prof.export_chrome_trace("od_report/fasterrcnn_resnet50_fpn_cpu_trace.json")
          break

      # # train for one epoch, printing every 10 iterations
      # train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
      # # update the learning rate
      # lr_scheduler.step()
      # # evaluate on the test dataset
      # evaluate(model, data_loader_test, device=device)
