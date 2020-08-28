import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

a = torch.ones(4)
b = torch.ones(4)
print("a", a)
print("b", b)

c = torch.dot(a, b)
print("# torch.dot（ベクトルの内積）:")
print(c)
print()

a = torch.ones((2, 3))
b = torch.ones((3, 4))
print("a", a)
print("b", b)

c = torch.matmul(a, b)
print("# torch.matmul（行列積）:")
print(c)
print(c.size())

a = torch.ones(2, 3, 4)
b = torch.ones(2, 4, 5)
print("a", a)
print("b", b)

print("# torch.bmm（batch matrix matrix product）:")
c = torch.bmm(a, b)
print(c)
print(c.size())

a = torch.ones((2, 3, 4))
b = torch.ones((2, 3))
print("a", a)
print("b", b)

c = torch.einsum('ijk,ij->k', (a, b))
print(c)

sum_c = torch.einsum('ijk,ij->', (a, b))
print(sum_c)
