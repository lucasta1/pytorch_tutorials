import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

a = torch.ones(1)
print(a)
print()

# GPUへの移動(すべて同じです)
b = a.cuda()
print(b)
b = a.to('cuda')
print(b)
b = torch.ones(1, device='cuda')
print(b)
print()

# CPUへの移動(すべて同じです)
c = b.cpu()
print(c)
c = b.to('cpu')
print(c)
c = torch.ones(1, device='cpu')
print(c)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.ones(1, device=device))