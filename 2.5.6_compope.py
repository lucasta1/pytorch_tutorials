import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

a = torch.arange(5)
print(a)
print()

print("# a >= 3:")
print(torch.ge(a, 3))  # greater than or equal to
print(a >= 3)  # pythonの比較演算子を用いることも可能
print()

print("# a > 3:")
print(torch.gt(a, 3))  # greater than
print(a > 3)  # pythonの比較演算子を用いることも可能
print()

print("# a <= 3:")
print(torch.le(a, 3))  # less than or equal to
print(a <= 3)  # pythonの比較演算子を用いることも可能
print()

print("# a < 3:")
print(torch.lt(a, 3))  # less than
print(a < 3)  # pythonの比較演算子を用いることも可能
print()

print("# a == 3:")
print(torch.eq(a, 3))  # equal to
print(a == 3)  # pythonの比較演算子を用いることも可能
print()