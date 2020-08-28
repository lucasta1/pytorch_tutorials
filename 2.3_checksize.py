import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

a = torch.ones((2, 3))

print(a.size())  # Tensorのサイズを取得
print(a.shape)  # .shapeでも可能

print(a.size(0))  # Tensorの0次元目のサイズを取得