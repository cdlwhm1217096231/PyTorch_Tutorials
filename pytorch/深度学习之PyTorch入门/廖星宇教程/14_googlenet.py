#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-25 09:07:53
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import torch
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
import torchvision.transforms as tfs
import numpy as np
from utils import train
import torch.nn as nn
import sys
sys.path.append("..")


"""inception模块"""


# 定义一个卷积和一个relu激活函数和一个batchnorm作为一个基本的层结构
def conv_relu(in_channel, out_channel, kernel, stride=1, padding=0):
    layer = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel, stride, padding),
        nn.BatchNorm2d(out_channel, eps=1e-3),
        nn.ReLU(True),
    )
    return layer


# inception模块
class inception(nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_5, out4_1):
        super(inception, self).__init__()

        # 第一条线路
        self.branch1x1 = conv_relu(in_channel, out1_1, 1)

        # 第二条线路
        self.branch3x3 = nn.Sequential(
            conv_relu(in_channel, out2_1, 1),
            conv_relu(out2_1, out2_3, 3, padding=1)
        )

        # 第三条线路
        self.branch5x5 = nn.Sequential(
            conv_relu(in_channel, out3_1, 1),
            conv_relu(out3_1, out3_5, 5, padding=2)
        )

        # 第四条线路
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            conv_relu(in_channel, out4_1, 1)
        )

    def forward(self, x):
        f1 = self.branch1x1(x)
        f2 = self.branch3x3(x)
        f3 = self.branch5x5(x)
        f4 = self.branch_pool(x)
        output = torch.cat((f1, f2, f3, f4), dim=1)
        return output


test_net = inception(3, 64, 48, 64, 64, 96, 32)
test_x = Variable(torch.zeros(1, 3, 96, 96))
print('input shape: {} x {} x {}'.format(
    test_x.shape[1], test_x.shape[2], test_x.shape[3]))
test_y = test_net(test_x)
print('output shape: {} x {} x {}'.format(
    test_y.shape[1], test_y.shape[2], test_y.shape[3]))


# 定义googlenet网络
class googlenet(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        super(googlenet, self).__init__()
        self.verbose = verbose

        self.block1 = nn.Sequential(
            conv_relu(in_channel, out_channel=64,
                      kernel=7, stride=2, padding=3),
            nn.MaxPool2d(3, 2)
        )

        self.block2 = nn.Sequential(
            conv_relu(64, 64, kernel=1),
            conv_relu(64, 192, kernel=3, padding=1),
            nn.MaxPool2d(3, 2)
        )

        self.block3 = nn.Sequential(
            inception(192, 64, 96, 128, 16, 32, 32),
            inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, 2)
        )

        self.block4 = nn.Sequential(
            inception(480, 192, 96, 208, 16, 48, 64),
            inception(512, 160, 112, 224, 24, 64, 64),
            inception(512, 128, 128, 256, 24, 64, 64),
            inception(512, 112, 144, 288, 32, 64, 64),
            inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2)
        )

        self.block5 = nn.Sequential(
            inception(832, 256, 160, 320, 32, 128, 128),
            inception(832, 384, 182, 384, 48, 128, 128),
            nn.AvgPool2d(2)
        )

        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.block1(x)
        if self.verbose:
            print('block 1 output: {}'.format(x.shape))
        x = self.block2(x)
        if self.verbose:
            print('block 2 output: {}'.format(x.shape))
        x = self.block3(x)
        if self.verbose:
            print('block 3 output: {}'.format(x.shape))
        x = self.block4(x)
        if self.verbose:
            print('block 4 output: {}'.format(x.shape))
        x = self.block5(x)
        if self.verbose:
            print('block 5 output: {}'.format(x.shape))
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


test_net = googlenet(3, 10, True)
test_x = Variable(torch.zeros(1, 3, 96, 96))  # （bs, n_c, n_h, n_w)
test_y = test_net(test_x)
print('output: {}'.format(test_y.shape))


# 数据预处理
def data_tf(x):
    x = x.resize((96, 96), 2)
    x = np.array(x, dtype="float32") / 255
    x = (x - 0.5) / 0.5
    x = x.transpose((2, 0, 1))  # 将channel放在第一个维度上
    x = torch.from_numpy(x)
    return x


# 数据预处理得到训练数据和测试数据
train_set = CIFAR10(root="./datasets/", train=True, transform=data_tf)
train_data = torch.utils.data.DataLoader(
    train_set, batch_size=64, shuffle=True)

test_set = CIFAR10(root="./datasets/", train=False, transform=data_tf)
test_data = torch.utils.data.DataLoader(
    test_set, batch_size=128, shuffle=False)

# 定义优化算法和损失函数
net = googlenet(3, 10)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()

# 训练网络
# train(net, train_data, test_data, num_epochs=20, optimizer, loss_func)
