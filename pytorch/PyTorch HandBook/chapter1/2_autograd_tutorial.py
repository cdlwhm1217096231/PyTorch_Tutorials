#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-02-25 21:31:00
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import torch
import numpy as np
import matplotlib.pyplot as plt

# 自动求导机制
"""
pytorch中所有的神经网络的核心是autograd包，autograd包为张量上的所有操作提供了自动求导功能。
"""

# 一、张量Tensor
"""
1.torch.Tensor是这个包的核心类，如果设置require_grad=True，那么将会追踪所有对于该张量的操作。当完成计算后通过调用
backward()，自动计算所有的梯度，这个张量的所有梯度将会自动积累的grad属性上。
2.要阻止张量跟踪历史记录，可以调用detach()方法将其与计算历史记录分离，并禁止跟踪它将来的计算记录。
3.为了防止跟踪历史记录，可以将代码块包装在with torch.no_grad()中，在评估模型时特别有用，因为模型可能具有requires_grad=True的可训练参数，但我们不需要梯度计算。
4.在自动梯度计算中还有另一个重要的类Function
5.Tensor和Function互相连接并生成一个非循环图，它表示和存储了完整的计算历史。每个张量都有一个grad_fn属性，这个属性引用一个创建了Tensor的Functions(除非这个张量是用户手动创建的，即这个张量的grad_fn=None)
6.如果需要计算导数，可以在Tensor上调用backward()。如果Tensor是一个标量(即是一个单一的数字)，则不需要为backward()指定任何参数，但是如果它由更多的元素，需要指定一个gradient参数来匹配张量的形状。
"""
# 注意:以前是将tensor包裹到Variable中提供自动梯度计算，Variable这个在0.4.1版本中已经被标注为过期了，现在可以直接使用Tensor
print("Pytorch版本:", torch.__version__)

# 创建一个张量并设置requires_grad=True
x = torch.ones(2, 2, requires_grad=True)
print("x: \n", x)
# 对张量进行操作
y = x + 2
print("y: \n", y)  # 结果y被计算出来了，grad_fn已经被自动生成了
print("y张量的属性: \n", y.grad_fn)

z = 3 * y * y
out = z.mean()
print("z: \n", z)
print("张量z的属性: \n", z.grad_fn)
print("out: \n", out)
print("张量out的属性: \n", out.grad_fn)

# requires_grad_(...)可以改变现有张量的requires_grad的属性，如果没能指定的话，默认输入的是requires_grad=False
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print("默认的requires_grad的属性值:", a.requires_grad)
a.requires_grad_(True)  # 使用替换操作in place来设置requires_grad的属性值
print("修改后的requires_grad值:", a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# 二. 梯度(gradient)
"""
反向传播是因为out是一个标量,out.backward()等价与out.backward(torch.tensor(1))
"""
out.backward()
print("x在1处的梯度是: \n", x.grad)  # d(out)/dx

# 更多的梯度操作
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print("y =", y)

gradient = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float)
y.backward(gradient)
print(x.grad)

# 如果requires_grad=True,但是又不希望进行autograd的计算，那么可以将变量包裹在with torch.no_grad()中:
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)
