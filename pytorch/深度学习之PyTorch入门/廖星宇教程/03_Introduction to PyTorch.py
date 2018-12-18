#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Version: python 3.5.2
# Tools: Pycharm 2017.2.2

import torchvision
import torch.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch

torch.manual_seed(1)   # 设置随机数种子，使得每次的结果是确定的

V_data = [1., 2., 3.]
v = torch.tensor(V_data)
print("v =", v)

# create a matrix
M_data = [[1., 2., 3.], [4., 5., 6.]]
m = torch.tensor(M_data)
print("m =\n", m)

# create a 3D tensor of size 2*2*2
T_data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
T = torch.tensor(T_data)
print("T = \n", T)
# index into V and get a scalar
print(v[0])
# get a python number from it
print(v[0].item())
# index into M and get a vector
print(m[0])
# index into T and get a matrix
print(T[0])
x = torch.randn((3, 4, 5))
print(x)
y = torch.LongTensor([1, 2, 3]).float()
print(y)
# Tensor 运算
x = torch.tensor([1, 2, 3]).float()
y = torch.tensor([4, 5, 6]).float()
z = x + y
print(z)
# 默认，it concatenates along the first axis(concatenates rows)
x_1 = torch.randn(2, 5)
y_1 = torch.randn(3, 5)
z_1 = torch.cat([x_1, y_1])
print(z_1)
# 拼接列
x_2 = torch.randn(2, 3)
y_2 = torch.randn(2, 5)
z_2 = torch.cat([x_2, y_2], 1)
print(z_2)
# reshape Tensor
x = torch.randn(2, 3, 4)
print(x)
print(x.view(2, 12))
# If one of the dimensions is -1, its size can be inferred
print(x.view(2, -1))
# computation graphs and automatic differentiation

# Tensor factory methods have a ``requires_grad`` flag
x = torch.tensor([1., 2., 3], requires_grad=True)
# With requires_grad=True, you can still do all the operations you previously
y = torch.tensor([4., 5., 6.], requires_grad=True)
z = x + y
print(z)
# But z knows something extra
print(z.grad_fn)
s = z.sum()
print(s)
print(s.grad_fn)
s.backward()
print(x.grad)
x = torch.randn(2, 2)
y = torch.randn(2, 2)
# By default, user created Tensors have ``requires_grad=False``
print(x.requires_grad, y.requires_grad)
z = x + y
# So you can't backprop through z
print(z.grad_fn)

# ``.requires_grad_( ... )`` changes an existing Tensor's ``requires_grad``
# flag in-place. The input flag defaults to ``True`` if not given.
x = x.requires_grad_()
y = y.requires_grad_()
# z contains enough information to compute gradients, as we saw above
z = x + y
print(z.grad_fn)
# If any input to an operation has ``requires_grad=True``, so will the output
print(z.requires_grad)

# Now z has the computation history that relates itself to x and y
# Can we just take its values, and **detach** it from its history?
new_z = z.detach()
print(new_z)
print(new_z.grad_fn)

# You can also stop autograd from tracking history on Tensors with .requires_grad``=True by wrapping the code block in ``with torch.no_grad():
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)
