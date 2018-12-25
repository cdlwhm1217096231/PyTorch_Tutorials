#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-25 08:35:13
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import sys

sys.path.append("..")

import torch
import torchvision
from torchvision.datasets import CIFAR10
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tfs
import numpy as np
from utils import train

"""
	resnet网络：解决梯度消失问题
"""


# 定义卷积层
def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)


# 定义单个的残差块
class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(residual_block, self).__init__()
        self.same_shape = same_shape
        stride = 1 if self.same_shape else 2

        self.conv1 = conv3x3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if not self.same_shape:
            self.conv3 = nn.Conv2d(
                in_channel, out_channel, kernel_size=1, stride=stride)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), True)
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(x + out, True)


# 输入输出形状相同
test_net = residual_block(32, 32)
test_x = Variable(torch.zeros(1, 32, 96, 96))
print('input: {}'.format(test_x.shape))
test_y = test_net(test_x)
print('output: {}'.format(test_y.shape))

# 输入输出形状不同
test_net = residual_block(3, 32, False)
test_x = Variable(torch.zeros(1, 3, 96, 96))
print('input: {}'.format(test_x.shape))
test_y = test_net(test_x)
print('output: {}'.format(test_y.shape))


# 将残差块堆叠起来，组成一个resnet网络
class resnet(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        super(resnet, self).__init__()
        self.verbose = verbose
        self.block1 = nn.Conv2d(in_channel, 64, 7, 2)
        self.block2 = nn.Sequential(
            nn.MaxPool2d(3, 2),
            residual_block(64, 64),
            residual_block(64, 64),
        )

        self.block3 = nn.Sequential(
            residual_block(64, 128, False),
            residual_block(128, 128),
        )

        self.block4 = nn.Sequential(
            residual_block(128, 256, False),
            residual_block(256, 256),
        )

        self.block5 = nn.Sequential(
            residual_block(256, 512, False),
            residual_block(512, 512),
            nn.AvgPool2d(3)
        )
        self.classifier = nn.Linear(512, out_features=num_classes)

    def forward(self, x):
        x = self.block1(x)
        if self.verbose:
            print("block1 output:{}".format(x.shape))

        x = self.block2(x)
        if self.verbose:
            print("block2 output:{}".format(x.shape))

        x = self.block3(x)
        if self.verbose:
            print("block3 output:{}".format(x.shape))

        x = self.block4(x)
        if self.verbose:
            print("block4 output:{}".format(x.shape))

        x = self.block5(x)
        if self.verbose:
            print("block5 output:{}".format(x.shape))

        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


test_net = resnet(3, 10, True)
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
net = resnet(3, 10)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()

# 训练网络
# train(net, train_data, test_data, num_epochs=20, optimizer, loss_func)
