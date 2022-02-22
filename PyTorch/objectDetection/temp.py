import torchvision
import torch
images = [torch.randn(3, 800, 1066).to("dml"), torch.randn(3, 1000, 800).to("dml")]
batched_imgs = torch.randn(2, 3, 1024, 1088).to("dml")
for img, pad_img in zip(images, batched_imgs):
    print (pad_img.shape, pad_img.dtype)
    print (img.shape, img.dtype)
    pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)