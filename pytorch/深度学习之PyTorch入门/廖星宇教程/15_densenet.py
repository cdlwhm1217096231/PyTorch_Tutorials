#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-25 09:31:58
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import torch
from torchvision.datasets import CIFAR10
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tfs
import torchvision
import numpy as np
from utils import train
from torch.autograd import Variable
import sys

sys.path.append("..")
"""
	densenet：与resnet不同的是，resnet是跨层求和，densenet是跨层将特征在通道维度上进行拼接
"""


# 定义一个卷积块，顺序是bn->relu->conv2d
def conv_block(in_channel, out_channel):
    layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel, out_channel,
                  kernel_size=3, padding=1, bias=False),
    )
    return layer


"""
	dense block 将每次的卷积的输出称为 growth_rate，因为如果输入是 in_channel，有 n 层，那么输出就是 in_channel + n * growh_rate
"""


class dense_block(nn.Module):
    def __init__(self, in_channel, growth_rate, num_layers):
        super(dense_block, self).__init__()
        block = []
        channel = in_channel
        for i in range(num_layers):
            block.append(conv_block(channel, growth_rate))
            channel += growth_rate

        self.net = nn.Sequential(*block)

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x


# 测试dense block
test_net = dense_block(3, 12, 3)
test_x = Variable(torch.zeros(1, 3, 96, 96))
print('input shape: {} x {} x {}'.format(
    test_x.shape[1], test_x.shape[2], test_x.shape[3]))
test_y = test_net(test_x)
print('output shape: {} x {} x {}'.format(
    test_y.shape[1], test_y.shape[2], test_y.shape[3]))


# 定义过渡层transition block
"""
	因为 DenseNet 会不断地对维度进行拼接， 所以当层数很高的时候，输出的通道数就会越来越大，参数和计算量也会越来越大，为了避免这个问题，需要引入过渡层将输出通道降低下来，同时也将输入的长宽减半，这个过渡层可以使用 1 x 1 的卷积
"""


def transition(in_channel, out_channel):
    trans_layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(True),
        nn.Conv2d(in_channel, out_channel, 1),
        nn.AvgPool2d(2, 2)
    )
    return trans_layer


# 验证过渡层是否正确
test_net = transition(3, 12)
test_x = Variable(torch.zeros(1, 3, 96, 96))
print('input shape: {} x {} x {}'.format(
    test_x.shape[1], test_x.shape[2], test_x.shape[3]))
test_y = test_net(test_x)
print('output shape: {} x {} x {}'.format(
    test_y.shape[1], test_y.shape[2], test_y.shape[3]))


# 定义densenet网络
class densenet(nn.Module):
    def __init__(self, in_channel, num_classes, growth_rate=32, block_layers=[6, 12, 24, 16]):
        super(densenet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1)
        )

        channels = 64
        block = []
        for i, layers in enumerate(block_layers):
            block.append(dense_block(channels, growth_rate, layers))
            channels += layers * growth_rate
            if i != len(block_layers) - 1:
                # 通过 transition 层将大小减半，通道数减半
                block.append(transition(channels, channels // 2))
                channels = channels // 2

        self.block2 = nn.Sequential(*block)
        self.block2.add_module('bn', nn.BatchNorm2d(channels))
        self.block2.add_module('relu', nn.ReLU(True))
        self.block2.add_module('avg_pool', nn.AvgPool2d(3))

        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


# 测试densenet网络
test_net = densenet(3, 10)
test_x = Variable(torch.zeros(1, 3, 96, 96))
test_y = test_net(test_x)
print('output: {}'.format(test_y.shape))


def data_tf(x):
    x = x.resize((96, 96), 2)
    x = np.array(x, dtype="float32") / 255
    x = (x - 0.5) / 0.5
    x = x.transpose((2, 0, 1))  # 将channel放在第一个维度上
    x = torch.from_numpy(x)
    return x


# 得到训练数据和测试数据
train_set = CIFAR10(root="./datasets/", train=True, transform=data_tf)
train_data = torch.utils.data.DataLoader(
    train_set, batch_size=64, shuffle=True)

test_set = CIFAR10(root="./datasets/", train=False, transform=data_tf)
test_data = torch.utils.data.DataLoader(
    test_set, batch_size=128, shuffle=False)

# 定义优化算法和损失函数
net = densenet(3, 10)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()

# 训练网络
# train(net, train_data, test_data, num_epochs=20, optimizer, loss_func)
