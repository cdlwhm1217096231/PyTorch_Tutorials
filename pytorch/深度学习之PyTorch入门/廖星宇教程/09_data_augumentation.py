#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-19 09:30:08
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from utils import train, resnet


"""
数据增强:
	1.对图片进行一定比例的缩放
	2.对图片进行随机位置的截取
	3.对图片进行随机的水平和竖直翻转
	4.对图片进行随机角度的翻转
	5.对图片进行亮度、对比度、颜色的随机变化
"""
im = Image.open("./datasets/cat.png")
# im.show()
# 随机比例缩放
print("缩放前shape:", im.size)
new_im = transforms.Resize((100, 200))(im)
print("缩放后shape:", new_im.size)
# new_im.show()

# 随机位置的截取
random_im1 = transforms.RandomCrop(100)(im)  # 随机剪裁出100*100的区域
# random_im1.show()

center_im = transforms.CenterCrop(100)(im)  # 从中心剪裁出100*100的区域
# center_im.show()

# 随机的水平和竖直方向的翻转
h_flip = transforms.RandomHorizontalFlip()(im)  # 随机水平翻转
# h_flip.show()
v_flip = transforms.RandomVerticalFlip()(im)  # 随机竖直翻转
# v_flip.show()

# 随机角度的翻转
rot_im = transforms.RandomRotation(45)(im)
# rot_im.show()

# 亮度、对比度和颜色的变化

# 亮度
bright_im = transforms.ColorJitter(brightness=1)(im)  # 随机从 0 ~ 2 之间亮度变化，1 表示原图
# bright_im.show()
# 对比度
contrast_im = transforms.ColorJitter(contrast=1)(im)  # 随机从 0 ~ 2 之间对比度变化，1 表示原图
# contrast_im.show()
# 颜色
color_im = transforms.ColorJitter(hue=0.5)(im)  # 随机从 -0.5 ~ 0.5 之间对颜色变化
# color_im.show()

# 将上面的方法综合起来进行使用
im_aug = transforms.Compose([transforms.Resize(120), transforms.RandomHorizontalFlip(
), transforms.RandomCrop(96), transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5)])


nrows = 3
ncols = 3
figsize = (8, 8)
_, figs = plt.subplots(nrows, ncols, figsize=figsize)
for i in range(nrows):
    for j in range(ncols):
        figs[i][j].imshow(im_aug(im))
        figs[i][j].axes.get_xaxis().set_visible(False)
        figs[i][j].axes.get_yaxis().set_visible(False)
# plt.show()

"""下面使用数据增强进行训练ResNet网络"""


# 使用数据增强进行数据预处理
def train_tf(x):
    im_aug = transforms.Compose([
        transforms.Resize(120),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(96),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    x = im_aug(x)
    return x


def test_tf(x):
    im_aug = transforms.Compose([
        transforms.Resize(96),
        transforms.ToTensor(),   # 转化成tensor
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])   # 归一化
    ])
    x = im_aug(x)
    return x


train_set = CIFAR10(root="./datasets/", train=True, download=True, transform=train_tf)
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_set = CIFAR10(root="./datasets/", train=False, download=True, transform=test_tf)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)

# 定义网络
net = resnet(3, 10)
# 定义损失函数和优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()
# 训练网络
train(net, train_data, test_data, 10, optimizer, loss_func)


# 不使用数据增强
def data_tf(x):
    im_aug = transforms.Compose([
        transforms.Resize(96),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    x = im_aug(x)
    return x
# 预处理数据
train_set = CIFAR10(root="./datasets/", train=True, download=True, transform=data_tf)
# 使用数据迭代器，获得每次的输入的batch
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
# 测试集数据与上述同理
test_set = CIFAR10(root="./datasets/", train=False, download=True, transform=data_tf)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)
# 定义网络
net = resnet(3, 10)
# 定义损失函数与优化算法
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()
# 训练网络
train(net, train_data, test_data, 10, optimizer, loss_func)
"""
总结：
    训练时:进行数据增强后，训练集的准确率会下降，训练难度增大
    测试时:进行数据增强后，训练的模型准确率会提高，增强模型的泛化能力
"""
