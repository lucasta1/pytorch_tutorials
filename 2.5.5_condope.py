import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

torch.manual_seed(34)

a = torch.randn((2, 3))
print(a)
print()

print("# torch.whereで0以下のmask:")
masked_a = torch.where(a > 0, torch.ones_like(a), torch.zeros_like(a))
print(masked_a)
print()

torch.manual_seed(34)

x = torch.randn((2, 3))
print(x)
print()

x_clipped = torch.clamp(x, 1e-10, 1e+10)
print("# torch.clampでクリッピング:")
print(x_clipped)
print(torch.log(x_clipped))
print()

print("# クリッピングしないとnanがでる:")
print(torch.log(x))