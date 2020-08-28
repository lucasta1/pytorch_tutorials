import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

# listを基にしたTensor作成
a = torch.tensor([[1, 2], [3, 4]])
print("# torch.tensor:")
print(a)
print()

# numpyを基にしたTensor作成
a = torch.tensor(np.array([[1, 2], [3, 4]]))
print("# torch.tensor:")
print(a)
print()

# 全要素が1の行列 (numpy.ones)
a = torch.ones((2, 3))
print("# torch.ones:")
print(a)
print()

# 全要素が0の行列 (numpy.ones)
a = torch.zeros((2, 3))
print("# torch.zeros:")
print(a)
print()

# 指定した値で満たされた行列 (numpy.full)
a = torch.full((2, 3), fill_value=99, dtype=torch.long)
print("# torch.full:")
print(a)
print()

# 単位行列 (numpy.eye)
a = torch.eye(2)
print("# torch.eye:")
print(a)
print()

torch.manual_seed(34)

# 標準正規分布 (numpy.random.randn)
print("# torch.randn（正規分布）:")
print(torch.randn(2))
print()

# [0, 1)の一様分布 (numpy.random.rand)
print("# torch.rand（一様分布）:")
print(torch.rand(2))
print()

# ベルヌーイ分布
probs = torch.rand((2, 3))
print("# 確率p:")
print(probs)
print()
print("# torch.bernoulli（ベルヌーイ分布）:")
print(torch.bernoulli(probs))
print()

# 多項分布 (np.random.multinomial)
probs = torch.tensor([0.2, 0.4, 0.4])
print("# torch.multinominal（多項分布）:")
print(torch.multinomial(probs, num_samples=10, replacement=True))