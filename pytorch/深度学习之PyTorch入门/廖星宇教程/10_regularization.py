#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-24 09:00:37
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

# L2正则化又称为权重衰减weight decay
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch import nn
from utils import train, resnet
from torchvision import transforms as tfs

import sys
sys.path.append("..")  # 对于自己写的模块和脚本不在同一目录下时，在脚本的开头添加此条语句


def data_tf(x):
    img_aug = tfs.Compose([
        tfs.Resize(96),
        tfs.ToTensor(),
        tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    x = img_aug(x)
    return x


train_set = CIFAR10(root="./datasets/", train=True, transform=data_tf)
train_data = torch.utils.data.DataLoader(
    train_set, batch_size=64, shuffle=True)

test_set = CIFAR10(root="./datasets/", train=False, transform=data_tf)
test_data = torch.utils.data.DataLoader(
    test_set, batch_size=128, shuffle=False)

net = resnet(3, 10)
# weight_deay是权重衰减参数，即正则化参数lambda
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay=1e-4)
loss_func = nn.CrossEntropyLoss()
train(net, train_data, test_data, 20, optimizer, loss_func)
