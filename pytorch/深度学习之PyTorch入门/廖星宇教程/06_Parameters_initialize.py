#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-18 16:12:04
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import numpy as np
import torch
from torch import nn


# 1.使用numpy初始化参数
net1 = nn.Sequential(
    nn.Linear(30, 40),
    nn.ReLU(),
    nn.Linear(40, 60),
    nn.ReLU(),
    nn.Linear(60, 10),
)

# 访问第一层参数
W1 = net1[0].weight
b1 = net1[0].bias
print(W1)  # 此时的参数W1实际上是一个Variable
print("获取Variable中的data:", W1.data)
# 1.使用随机生成数的方法对权重进行初始化
net1[0].weight.data = torch.from_numpy(np.random.uniform(3, 5, size=(40, 30)))
print(net1[0].weight)
# 模型中相同类型的层都需要初始化成相同的方式
for layer in net1:
    if isinstance(layer, nn.Linear):
        param_shape = layer.weight.shape
        layer.weight.data = torch.from_numpy(
            np.random.normal(0, 0.5, size=param_shape))
        # 定义为均值是0，方差是0.5的正态分布


# 2.非常流行的一种初始化方法Xavier


class sim_net(nn.Module):
    def __init__(self):
        super(sim_net, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(30, 40),
            nn.ReLU(),
        )
        self.l1[0].weight.data = torch.randn(40, 30)  # 直接对某一层参数初始化

        self.l2 = nn.Sequential(
            nn.Linear(40, 50),
            nn.ReLU(),
        )

        self.l3 = nn.Sequential(
            nn.Linear(50, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x


net2 = sim_net()
# 访问children----只会访问到三个Sequential
for i in net2.children():
    print(i)
print("*" * 60)
# 访问module------不仅访问到Squential，还会访问到Squential的内部
for i in net2.modules():
    print(i)
print("*" * 60)
for layer in net2.modules():
    if isinstance(layer, nn.Linear):
        param_shape = layer.weight.shape
        layer.weight.data = torch.from_numpy(
            np.random.normal(0, 0.5, size=param_shape))

# 3.torch.nn.init
from torch.nn import init

print("修改前:", net1[0].weight)
init.xavier_uniform_(net1[0].weight)  # 即第二种方法，只是pytorch中内置的
print("修改后:", net1[0].weight)
