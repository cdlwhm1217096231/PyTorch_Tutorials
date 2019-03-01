#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-02-28 16:12:01
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader   # utils是工具集的意思
import numpy as np


"""
1.torch.nn是专门为神经网络设计的模块化接口，nn构建在autograd上，可以用来定义和运行神经网络。
2.约定：torch.nn设置别名nn
3.nn.functional这个包中包含了神经网络中使用的一些常用函数,一般把nn.functional设置别名为F
"""

# 一、定义一个网络
"""
pytorch已经为我们准备好了现成的网络模型，只要继承nn.Module,并实现它的forward()方法即可，pytorch会自动进行求导，实现反向传播backward()。在forward()函数中可以使用任何tensor支持的函数，还可以使用if for循环 print log等python语法
"""


class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行分类的构造函数
        super(Net, self).__init__()
        # 卷积层1表示输入的是单通道的图片，6表示输出通道数，3表示卷积核3*3
        self.conv1 = nn.Conv2d(1, 6, 3)
        # 线性层输入是1350个特征，输出是10个特征
        self.fc1 = nn.Linear(1350, 10)  # 1350个特征是如何计算得到的，要看后面的forward()函数

    def forward(self, x):
        print(x.size())  # 结果是[1, 1, 32, 32]
        # 卷积--->激活---->池化
        x = self.conv1(x)  # 根据卷积的尺寸计算公式，计算结果是30
        x = F.relu(x)
        print(x.size())  # 结果是[1, 6, 30, 30]
        x = F.max_pool2d(x, (2, 2))  # 池化层后，计算的结果是15
        x = F.relu(x)
        print(x.size())  # 结果是[1, 6, 15, 15]
        # flatten操作，将[1,6,15,15]变为[1,1350]
        x = x.view(1, -1)
        print(x.size())  # 这里就是fc1层的输入1350
        x = self.fc1(x)
        return x


net = Net()
print(net)

# 网络可学习的参数，通过net.parameters()返回
for parameters in net.parameters():
    print(parameters)

# net.named_parameters可同时返回可学习的参数及名称
for name, parameters in net.named_parameters():
    print(name, ":", parameters.size())

# forward()函数的输入和输出都是Tensor
inputs = torch.randn(1, 1, 32, 32)
output = net(inputs)
print(output.size())
print(inputs.size())

# 反向传播前，必须将所有参数的梯度清零
net.zero_grad()
output.backward(torch.ones(1, 10))  # 反向传播自动实现
"""
注意：torch.nn只支持mini-batches，不支持一次只输入一个样本，即一次必须是一个batch
即使输入的是一个样本，也会对样本进行分组。所以，所有的输入都会增加一个维度
"""

# 损失函数
targets = torch.arange(0, 10).view(1, 10).float()
criterion = nn.MSELoss()
loss = criterion(output, targets)
print(loss.item())

# 优化器
"""
在反向传播计算完所有参数的梯度后，还需使用优化方法来更新网络的权重和参数，如SGD的更新策略w = w - learing_rate * gradient,在torch.optim中实现了大多数的优化方法，如RMSProp、Adam、SGD等
"""
output = net(inputs)
criterion = nn.MSELoss()
loss = criterion(output, targets)
# 新建一个优化器，SGD只需要调整参数和学习率
optimizer = optim.SGD(net.parameters(), lr=1e-3)
# 先梯度清零----下面的语句与net.zero_grad()效果一样
optimizer.zero_grad()
loss.backward()
optimizer.step()  # 更新参数
