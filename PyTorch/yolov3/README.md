# YOLO V3 Object Detection

Sample scripts for training and inferencing of the YOLO V3 object detection model using Pytorch on DirectML.

This is a fork of https://github.com/ultralytics/yolov3 at commit [1be3170](https://github.com/ultralytics/yolov3/commit/1be31704c9c690929e4f6e6d950f40755ef2dcdc). The original code is here under the terms of the GNU GENERAL PUBLIC LICENSE. See [LICENSE](./LICENSE) for more information.

- [YOLO V3 Object Detection](#yolo-v3-object-detection)
  - [Setup](#setup)
  - [Training](#training)
  - [Predict](#predict)

## Setup
Install the following prerequisites:
```
pip install -r pytorch\yolov3\requirements.txt 
```

## Training

A helper script exists to train yolov3 with default data, batch size, and so on, and testing is enabled by default, to disable, use --notest:

```
python train.py --batch-size 4 --device dml --nosave --notest
```

The first few lines of output should look similar to the following (exact numbers may change):

```
github: skipping check (not a git repository)
Namespace(adam=False, artifact_alias='latest', batch_size=4, bbox_interval=-1, bucket='', cache_images=False, cfg='.\\models\\yolov3.yaml', data='.\\data\\coco128.yaml', device='dml', entity=None, epochs=300, evolve=False, exist_ok=False, global_rank=-1, hyp='data/hyp.scratch.yaml', image_weights=False, img_size=[640, 640], label_smoothing=0.0, linear_lr=False, local_rank=-1, multi_scale=False, name='exp', noautoanchor=False, nosave=False, notest=False, project='runs/train', quad=False, rect=False, resume=False, save_dir='runs\\train\\exp2', save_period=-1, single_cls=False, sync_bn=False, total_batch_size=4, upload_dataset=False, weights='yolov3.pt', workers=8, world_size=1)
tensorboard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
hyperparameters: lr0=0.01, lrf=0.2, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0
wandb: Install Weights & Biases for YOLOv3 logging with 'pip install wandb' (recommended)

                 from  n    params  module                                  arguments
  0                -1  1       928  models.common.Conv                      [3, 32, 3, 1]
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]
  2                -1  1     20672  models.common.Bottleneck                [64, 64]
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]
  4                -1  2    164608  models.common.Bottleneck                [128, 128]
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]
  6                -1  8   2627584  models.common.Bottleneck                [256, 256]
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]
  8                -1  8  10498048  models.common.Bottleneck                [512, 512]
  9                -1  1   4720640  models.common.Conv                      [512, 1024, 3, 2]
 10                -1  4  20983808  models.common.Bottleneck                [1024, 1024]
 11                -1  1   5245952  models.common.Bottleneck                [1024, 1024, False]
 12                -1  1    525312  models.common.Conv                      [1024, 512, [1, 1]]
 13                -1  1   4720640  models.common.Conv                      [512, 1024, 3, 1]
 14                -1  1    525312  models.common.Conv                      [1024, 512, 1, 1]
 15                -1  1   4720640  models.common.Conv                      [512, 1024, 3, 1]
 16                -2  1    131584  models.common.Conv                      [512, 256, 1, 1]
 17                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']
 18           [-1, 8]  1         0  models.common.Concat                    [1]
 19                -1  1   1377792  models.common.Bottleneck                [768, 512, False]
 20                -1  1   1312256  models.common.Bottleneck                [512, 512, False]
 21                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]
 22                -1  1   1180672  models.common.Conv                      [256, 512, 3, 1]
 23                -2  1     33024  models.common.Conv                      [256, 128, 1, 1]
 24                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']
 25           [-1, 6]  1         0  models.common.Concat                    [1]
 26                -1  1    344832  models.common.Bottleneck                [384, 256, False]
 27                -1  2    656896  models.common.Bottleneck                [256, 256, False]
 28      [27, 22, 15]  1    457725  models.yolo.Detect                      [80, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [256, 512, 1024]]
Model Summary: 333 layers, 61949149 parameters, 61949149 gradients, 156.4 GFLOPS

Transferred 438/440 items from yolov3.pt
Scaled weight_decay = 0.0005
Optimizer groups: 75 .bias, 75 conv.weight, 72 other
train: Scanning '..\coco128\labels\train2017.cache' images and labels... 128 found, 0 missing, 2 empty, 0 corrupted: 100%|█| 128/128 [00:00<?,
val: Scanning '..\coco128\labels\train2017.cache' images and labels... 128 found, 0 missing, 2 empty, 0 corrupted: 100%|█| 128/128 [00:00<?, ?i
Plotting labels...

autoanchor: Analyzing anchors... anchors/target = 4.26, Best Possible Recall (BPR) = 0.9946
Image sizes 640 train, 640 test
Using 4 dataloader workers
Logging results to runs\train\exp2
Starting training for 300 epochs...

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     0/299        0G   0.02813   0.02754  0.007858   0.06353        72       640:   0%|                                 | 0/32 [00:04<?, ?it/s]E:\Anaconda3\envs\yolov3\lib\site-packages\torch\jit\_trace.py:727: UserWarning: The input to trace is already a ScriptModule, tracing it is a no-op. Returning the object as is.
  warnings.warn(
     0/299        0G   0.03181    0.0266   0.01114   0.06955        53       640: 100%|████████████████████████| 32/32 [02:10<00:00,  4.09s/it]
               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100%|████████████| 16/16 [01:40<00:00,  6.26s/it]
                 all         128         929       0.717       0.678       0.774       0.541
```


## Predict

Once the model is trained and saved we can now test the model using the following steps.

You can specified the model by using --weight

```
python .\detect.py --weights yolov3.pt --device dml
```

You should see the result such as this:

```
(pytorch-dml) PS E:\xianz\ultralytics-yolov3-dml> python .\detect.py --source data/images --device dml
Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.25, device='dml', exist_ok=False, hide_conf=False, hide_labels=False, img_size=640, iou_thres=0.45, line_thickness=3, max_det=1000, name='exp', nosave=False, project='runs/detect', save_conf=False, save_crop=False, save_txt=False, source='data/images', update=False, view_img=False, weights='yolov3.pt')
Fusing layers...
Model Summary: 261 layers, 61922845 parameters, 0 gradients
image 1/2 F:\xianz\DirectML\PyTorch\yolov3\data\images\bus.jpg: 640x480 4 persons, 1 bus, Done. (0.285s)
image 2/2 F:\xianz\DirectML\PyTorch\yolov3\data\images\zidane.jpg: 384x640 2 persons, 3 ties, Done. (0.170s)
Results saved to runs\detect\exp3
Done. (0.735s)
```
