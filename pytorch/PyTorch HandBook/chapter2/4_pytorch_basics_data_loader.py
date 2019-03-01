#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2019-03-01 10:02:11
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

# 1.pytorch基础：数据的加载与预处理
"""
pytorch通过torch.utils.data对一般常用的数据进行了封装，可以很容易实现对线程数据预读和批量加载
torchvision已经预先实现了常用图像的数据集，包括cifar10 imagenet coco mnist lsun等数据集，可以通过torchvision.datasets来进行调用。
"""

# 1.1 DataSet类来自torch.utils.data中
"""
为了能方便读取，需要将使用的数据包封装为DataSet类，自定义DataSet时，需要继承该类并重写两个成员方法：
    __getitem()__：定义了每次怎么读数据
    __len()__:定义了自定义数据集的大小
"""


class BulldozerDataSet(Dataset):
    def __init__(self, cvs_file):
        # 实现初始化方法，在初始化时将数据进行加载
        self.df = pd.read_csv(cvs_file)

    def __len__(self):
        # 返回df的长度
        return len(df)

    def __getitem(self, idx):
        # 根据索引，返回一列数据
        return self.df.iloc[idx].SalePrice


# 数据集定义已经完成，可以实例化一个对象进行访问
ds_demo = BulldozerDataSet("median_benchmark.csv")
# 由于实现了__len__()方法可以直接使用len获取数据总数
print(len(ds_demo))
# 用索引可以直接访问对应的数据
print(ds_demo[0])


# 1.2 DataLoader
"""
DataLoader提供了对DataSet的读取操作，常用的参数有:batch_size(每个batch的大小)、shuffle(是否打乱)、num_workers(加载数据的时候使用几个子线程)
"""
dl = DataLoader(ds_demo, batch_size=10, shuffle=True, num_workers=0)

# !!! DataLoader返回的是一个迭代器，可以使用迭代器分次获取数据
iter_data = iter(dl)
print(next(iter_data))
# 或者使用for循环来对其进行遍历
for i, data in enumerate(dl):
    print(i, '--->', data)
    break

# 1.3 torchvision包
"""
torchvision是pytorch专门用来处理图像的库
"""

# 1.3.1 torchvision.datasets
"""
可以理解为pytorch团队自定义的dataset，拿来就可以使用：CIFAR10、COCO、MNIST等数据集
"""
# train_set = datasets.MNIST(root="./data", train=True,
# download=True, transform=None)
"""
root:表示MNIST数据集的加载目录
train:表示是否加载数据库的训练集，false时加载测试集
download:表示是否自动下载MNIST数据集
transform:表示是否需要对数据进行预处理，none表示不进行预处理
"""

# 1.3.2 torchvision.models
"""
torchvision不仅提供常用的图片数据集，还提供训练好的模型，可以在加载后使用，或者在进行迁移学习时使用
torchvision.models模块的子模块中包含以下模型结构： AlexNet VGG ResNet SqueezeNet DenseNet
"""

# resnet18 = models.resnet18(pretrained=True)


# 1.3.3 trochvision.transforms
"""
transforms模块提供了一般的图像转换操作类，用于数据的处理和增广
"""
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.RandomRotation(-45, 45),  # 随机旋转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.229, 0.224, 0.225))  # R,G,B每层的归一化用到的均值和方差
])
