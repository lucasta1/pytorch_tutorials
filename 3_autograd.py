import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

# 順伝播の計算
x = torch.randn(4, 4)
y = torch.randn(4, 1)

w = torch.randn(4, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

y_pred = torch.matmul(x, w) + b

# 目的関数の定義
loss = (y_pred - y).pow(2).sum()

# ユーザが作成したTensorはgrad_fn=None
print(x.grad_fn)
print(y.grad_fn)
print(w.grad_fn)
print(b.grad_fn)
print()

# Functionによって計算されたTensorはgrad_fnを有する
print(y_pred.grad_fn)

# まだ勾配は計算されていない
print(x.grad)
print(y.grad)
print(w.grad)
print(b.grad)

# 逆伝播
loss.backward()

# requires_grad=Trueを指定した変数は勾配が計算されている
print(x.grad)
print(y.grad)
print(w.grad)
print(b.grad)

x = torch.randn(4, 4)
y = torch.randn(4, 1)

w = torch.randn(4, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
b = b.detach()  # bの勾配計算を停止

y_pred = torch.matmul(x, w) + b

loss = (y_pred - y).pow(2).sum()

loss.backward()

print(w.grad)  # 勾配を有する
print(b.grad)  # 勾配を有さない

with torch.no_grad():
    y_eval = torch.matmul(x, w) + b  # y_predと同様の計算を行う

print('requires_grad of y_pred:', y_pred.requires_grad)  # requires_grad=True
print('requires_grad of y_eval:', y_eval.requires_grad)  # requires_grad=False