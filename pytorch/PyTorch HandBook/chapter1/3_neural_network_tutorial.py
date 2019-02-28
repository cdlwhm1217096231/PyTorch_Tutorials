#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-02-26 10:41:22
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
"""
1.使用torch.nn包来创建神经网络，nn包依赖autograd包来定义模型并求导。一个nn.Module包含各个层和一个forward(input)的方法，该方法返回output。
2.神经网络的典型训练过程如下：
    a.定义包含一些可以学习的参数(或叫权重)的神经网络
    b.在数据集上迭代
    c.通过神经网络处理
    d.计算损失(输出结果和正确值的差值大小)
    e.将梯度反向传播回网络的参数
    f.更新网络的参数，主要使用如下简单的规则: weight = weight - learning_rate * gradient
"""

# a~c.


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel,6 output channels,5*5 filters
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # fully connect layer
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # max pooling over a (2,2) windows
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # if the size is square,you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        print("size:", size)
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

# 在模型中必须要定义forward函数，backward函数会被autograd自动创建。可以在forward函数中使用任何针对tensor的操作
# net.parameters():返回可以被学习的参数(权重)列表和值

params = list(net.parameters())
print("parameters: \n", params)
print("list length is:", len(params))
print("conv1 weight is:", params[0].size())

# 测试随机输入32*32
input = torch.randn(1, 1, 32, 32)
out = net(input)
print("out: \n", out)

# 然后将所有的参数梯度缓存清零，再进行随机梯度的反向传播，必须先清零！
net.zero_grad()
out.backward(torch.randn(1, 10))

"""
注：torch.nn只支持小批量的输入。整个torch.nn包都支持小批量样本，不支持单个样本。例如，nn.Conv2d接收一个4维的张量,每一维分别是samples * n_channel * n_h * n_w(样本数,通道数,高度,宽度)。如果有单个样本，只需使用input.unsqueeze(0)来添加其他的维数。
"""

# d.损失函数:一个损失函数接收一对(output,target)作为输入，计算一个值来估计网络的输出和目标值相差多少
output = net(input)
target = torch.randn(10)  # 随机值作为期望输出
target = target.view(1, -1)  # 使得期望值与output的shape相同
criterion = nn.MSELoss()
loss = criterion(output, target)
print("loss =", loss)
"""
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
"""
print("MSELoss:", loss.grad_fn)
print("Linear:", loss.grad_fn.next_functions[0][0])
print("ReLU:", loss.grad_fn.next_functions[0][0].next_functions[0][0])

# e.反向传播
"""
调用loss.backward()获得反向传播的误差,在调用前需要清除已存在的梯度，否则梯度将会被累加到已存在的梯度上！
"""
net.zero_grad()  # 清除梯度
print("conv1.bias.grad before backward:", net.conv1.bias.grad)
loss.backward()
print("conv1.bias.grad after backward:", net.conv1.bias.grad)

# f.更新权重
"""
在实践中，最简单的权重更新规则是随机梯度下降SGD: weight = weight - learning_rate * gradient
可以使用python代码实现这个简单的规则：
    learning_rate= 0.01
    for f in net.parameters():
        f.data.sub_(f.grad.data * learning_rate)
但是当使用神经网络要想实现各种不同的更新规则时,如SGD，Adam，RMSPROP等，pytorch中构建了一个包torch.optim实现了所有的规则
"""
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop
optimizer.zero_grad()  # zero the gradient buffers!!!,一定不要少了
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()  # Does the update
