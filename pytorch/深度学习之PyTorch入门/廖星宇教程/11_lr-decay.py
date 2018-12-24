#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-24 10:20:13
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms as tfs
from datetime import datetime
from utils import resnet
import matplotlib.pyplot as plt


import sys
sys.path.append("..")

"""
	学习率衰减：在开始学习时。学习率可以设置大一些，到loss下降到一定的程度后，将要减小这个学习率
"""


net = resnet(3, 10)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay=1e-4)
# 通过optimizer.param_groups来得到所有的参数和其对应的属性,实质是一个字典
print("learning rate:", optimizer.param_groups[0]["lr"])
print("weight decay:", optimizer.param_groups[0]["weight_decay"])
# 通过这个属性可以修改许多属性
optimizer.param_groups[0]["lr"] = 1e-5
optimizer.param_groups[0]["weight_decay"] = 0.1
print("learning rate:", optimizer.param_groups[0]["lr"])
print("weight decay:", optimizer.param_groups[0]["weight_decay"])
# 为了防止有多个参数组,可以使用循环来实现
for param_group in optimizer.param_groups:
    param_group["lr"] = 1e-1


# 设置学习率
def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# 使用数据增强
def train_tf(x):
    img_aug = tfs.Compose([
        tfs.Resize(120),
        tfs.RandomHorizontalFlip(),
        tfs.RandomCrop(96),
        tfs.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        tfs.ToTensor(),
        tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    x = img_aug(x)
    return x


def test_tf(x):
    img_aug = tfs.Compose([
        tfs.Resize(120),
        tfs.RandomHorizontalFlip(),
        tfs.RandomCrop(96),
        tfs.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        tfs.ToTensor(),
        tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    x = img_aug(x)
    return x


train_set = CIFAR10(root="./datasets/", train=True, transform=train_tf)
train_data = torch.utils.data.DataLoader(
    train_set, batch_size=256, shuffle=True)

test_set = CIFAR10(root="./datasets/", train=False, transform=test_tf)
test_data = torch.utils.data.DataLoader(
    test_set, batch_size=256, shuffle=False)

net = resnet(3, 10)
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=1e-4)
loss_func = nn.CrossEntropyLoss()


train_losses = []
test_losses = []
if torch.cuda.is_available():
    net = net.cuda()
prev_time = datetime.now()
for epoch in range(30):
    if epoch == 20:
        set_learning_rate(optimizer, 0.01)   # 20次修改学习率是0.01
    train_loss = 0
    net = net.train()
    for im, label in train_data:
        if torch.cuda.is_available():
            im = Variable(im.cuda())   # (bs, 3, h, w)
            label = Variable(label.cuda())   # (bs, h, w)
        else:
            im = Variable(im)
            label = Variable(label)
        # 前向传播
        output = net(im)
        loss = loss_func(output, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = "Time %02d:%02d:%02d" % (h, m, s)
    test_loss = 0
    test_acc = 0
    net = net.eval()
    for im, label in test_data:
        if torch.cuda.is_available():
            im = Variable(im.cuda(), volatile=True)
            lable = Variable(label.cuda(), volatile=True)
        else:
            im = Variable(im, volatile=True)
            lable = Variable(label, volatile=True)
        output = net(im)
        loss = loss_func(output, label)
        test_loss += loss.item()
    epoch_str = ("Epoch %d. Train Loss： %f,Valid Loss: %f, " % (
        epoch, train_loss / len(train_data), test_loss / len(test_data)))
    prev_time = cur_time
    train_losses.append(train_loss / len(train_data))
    test_losses.append(test_loss / len(test_data))
    print(epoch_str + time_str)


# 画图
plt.plot(train_losses, label="train")
plt.plot(test_losses, label="test")
plt.xlabel('epoch')
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()
