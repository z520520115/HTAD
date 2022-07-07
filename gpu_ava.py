import torch

print(torch.__version__)
print(torch.backends.cudnn.enabled)
print(torch.cuda.is_available())
print(torch.version.cuda)