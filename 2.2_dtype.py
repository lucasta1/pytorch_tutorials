import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
print(a.dtype)
print(a)
print()

a = torch.ones((2, 3), dtype=torch.int)
print(a.dtype)
print(a)
print()

a = torch.ones((2, 3), dtype=torch.long)
print(a.dtype)
print(a)
print()