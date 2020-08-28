import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

a = torch.ones((2, 3))
print(a)
print()

# numpy.ndarrayへの変換
print("# a.numpy():")
print(a.numpy())
print()

# listへの変換
print("# a.tolist():")
print(a.tolist())
print()

# scalarへの変換には.item()を使います
print("# a.item():")
print(a.sum().item())