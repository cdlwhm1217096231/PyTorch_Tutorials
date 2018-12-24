#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-24 16:26:31
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.functional as F
from torchvision.datasets import CIFAR10
import torchvision.transforms as tfs
import numpy as np
from utils import train
import sys
sys.path.append("..")

"""vgg网络"""
# cifar10数据集介绍:50000张训练图片，10000张测试图片，图片是32*32*3，10分类问题


def vgg_block(num_convs, in_channels, out_channels):
    """
            num_convs:模型的层数
            in_channels:输入的通道数
            out_channels:输出的通道数
    """
    net = [nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, padding=1), nn.ReLU(True)]
    for i in range(num_convs - 1):
        net.append(nn.Conv2d(out_channels, out_channels,
                             kernel_size=3, padding=1))
        net.append(nn.ReLU(True))
    net.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*net)


block_demo = vgg_block(3, 64, 128)  # (num_convs, in_channels, out_channels)
print(block_demo)


# 定义输入(1, 64, 300, 300)  (bs, nc, nh, nw)
input_demo = Variable(torch.zeros(1, 64, 300, 300))
output_demo = block_demo(input_demo)
print(output_demo.shape)


# 定义一个函数将vgg block进行堆叠
def vgg_stack(num_convs, channels):
    net = []
    for n, c in zip(num_convs, channels):
        in_c = c[0]
        out_c = c[1]
        net.append(vgg_block(n, in_c, out_c))
    return nn.Sequential(*net)


vgg_net = vgg_stack((1, 1, 2, 2, 2), ((3, 64), (64, 128),
                                      (128, 256), (256, 512), (512, 512)))
print(vgg_net)
# 测试池化层效果
test_x = Variable(torch.zeros(1, 3, 256, 256))
test_y = vgg_net(test_x)
print(test_y.shape)


# 定义网络
class vgg(nn.Module):
    def __init__(self):
        super(vgg, self).__init__()
        self.feature = vgg_net
        self.fc = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(True),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


# 数据预处理
def data_tf(x):
    x = np.array(x, dtype="float32") / 255
    x = (x - 0.5) / 0.5
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy()
    return x


# 得到训练数据和测试数据
train_set = CIFAR10(root="./datasets/", train=True, transform=data_tf)
train_data = torch.utils.data.DataLoader(
    train_set, batch_size=64, shuffle=True)

test_set = CIFAR10(root="./datasets/", train=False, transform=data_tf)
test_data = torch.utils.data.DataLoader(
    test_set, batch_size=128, shuffle=False)

# 定义优化算法和损失函数
net = vgg()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-1)
loss_func = nn.CrossEntropyLoss()

# 训练网络
train(net, train_data, test_data, num_epochs=20, optimizer, loss_func)
