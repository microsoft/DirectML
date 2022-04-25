import torch
import torch.autograd.profiler as profiler
import torchvision
import argparse

device = torch.device('dml')

object_detection_model_list = [
  'fasterrcnn_resnet50_fpn',
  'fasterrcnn_mobilenet_v3_large_fpn',
  'fasterrcnn_mobilenet_v3_large_320_fpn',
  'retinanet_resnet50_fpn',
  'maskrcnn_resnet50_fpn',
]

def get_model(model_str):
  if model_str.lower() == 'fasterrcnn_resnet50_fpn':
    return torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
  elif model_str.lower() == 'fasterrcnn_mobilenet_v3_large_fpn':
    return torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
  elif model_str.lower() == 'fasterrcnn_mobilenet_v3_large_320_fpn':
    return torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
  elif model_str.lower() == 'retinanet_resnet50_fpn':
    return torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
  elif model_str.lower() == 'maskrcnn_resnet50_fpn':
    return torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
  else:
        raise Exception(f"Model {model_str} is not supported yet!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model', type=str, default='fasterrcnn_resnet50_fpn', help='The model to use.')
    parser.add_argument('--device', type=str, default='dml', help='The device to use for training.')
    parser.add_argument('--save_trace', type=bool, default=False, help='Trace performance.')

    args = parser.parse_args()

    model = get_model(args.model).to(args.device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    model.train()

    # generate garbage data for training one iteration
    images, boxes = torch.rand(1, 3, 600, 1200).to(device), torch.rand(1, 11, 4).sort().values.to(device)
    masks = torch.randint(0, 2, (1, 3, 600, 1200)).bool().to(device)
    labels = torch.randint(1, 91, (1, 11)).to(device)
    images = list(image for image in images)
    targets = []

    for i in range(len(images)):
      d = {}
      d['boxes'] = boxes[i]
      d['labels'] = labels[i]
      d['masks'] = masks[i]
      targets.append(d)

    with profiler.profile(record_shapes=True, with_stack=True, profile_memory=True) as prof:
        with profiler.record_function("model_inference"):
          loss_dict = model(images, targets)
          losses = sum(loss for loss in loss_dict.values())
          optimizer.zero_grad()
          losses.backward()
          optimizer.step()

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=1000))
    if args.save_trace:
      trace_path = '_'.join(["train", args.model, "dml", "trace.json"])
      prof.export_chrome_trace(trace_path)
