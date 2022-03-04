import torch
import torch.autograd.profiler as profiler

a = torch.tensor([1,2,1,1], dtype=torch.float32).to("dml").requires_grad_(True)
output = torch.where(a==1)[0]
output.sum().backward()

print (output[0].sum())


