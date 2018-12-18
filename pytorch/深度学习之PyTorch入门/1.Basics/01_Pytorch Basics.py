#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-06 15:07:42
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim


# 1.Basic autograd example1
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)
# 构建计算图
y = w * x + b   # y = 2 * x + 3
# 计算梯度
y.backward()
# 打印梯度
print(x.grad)
print(w.grad)
print(b.grad)
# 2. Basic autograd exampple2
x = torch.randn(10, 3)
y = torch.randn(10, 2)
# 构建全连接层
linear = nn.Linear(3, 2)
print("w:", linear.weight)
print("b:", linear.bias)
# 定义损失函数与优化算法
loss_func = nn.MSELoss()
optimizer = optim.SGD(linear.parameters(), lr=0.01)
# 前向传播
y_hat = linear(x)
# 计算loss
loss = loss_func(y_hat, y)
print("loss:", loss.item())
# 反向传播
loss.backward()
# 打印梯度
print("dL/dw:", linear.weight.grad)
print("dL/db:", linear.bias.grad)
# 进行一次梯度下降
optimizer.step()
"""
# 可以等价于下面的语句
linear.weight.data.sub_(0.01 * linear.weight.data)
linear.bias.data.sub_(0.01 *linear.bias.data)
"""
# 经过一次梯度下降后的loss值
y_hat = linear(x)
loss = loss_func(y_hat, y)
print("经过一次梯度下降后的loss值:", loss.item())
# 3.Loading data from numpy
# 创建一个numpy array
x = np.array([[1, 2], [3, 4]])
# 将numpy中的array转为tensor
y = torch.from_numpy(x)
print("tensor y:", y)
# 将tensor转为numpy中的array
z = y.numpy()
print("array z:", z)
# 4.Input pipline
# 下载和构建CIFAR-10数据集
train_dataset = torchvision.datasets.CIFAR10(
    root="../datasets/", train=True, transform=transforms.ToTensor(), download=True)
image, label = train_dataset[0]
print(image.size())
print(label)
# Data Loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True)
# 当迭代开始时，用队列和线程开始从文件中读取数据
data_iter = iter(train_loader)   # 迭代器
# Mini-batch images and labels
images, labels = data_iter.next()
# 实际中，这样使用
for images, labels in train_loader:
    # 此处写训练代码
    pass
# 5.Input pipline for custom dataset
# 使用下面的代码构建自定义数据集


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # TODO
        # 初始化文件路径或列出文件名字
        pass

    def __getitem__(self, index):
       # 从文件读取数据（np.fromfile, PIL.Image.open）
       # 预处理数据(torchvision.Transfrom)
       # 返回一个数据(images, labels)
        pass

    def __len__(self):
        return 0


customdataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(
    dataset=customdataset, batch_size=64, shuffle=True)
# 6.Pretrained model
# 下载预训练模型Resnet-18
resnet = torchvision.models.resnet18(pretrained=True)
# 如果只是微调模型顶层
for param in resnet.parameters():
    param.requires_grad = True
# 替换顶层进行微调
resnet.fc = nn.Linear(resnet.fc.in_features, 100)
# 前向传播
images = torch.randn(64, 3, 224, 224)
output = resnet(images)
print(output.size())
# 7.save and load the model
# 保存整个模型
torch.save(resnet, "model.ckpt")
model = torch.load("model.ckpt")
# 只保存模型的参数----推荐
torch.save(resnet.state_dict(), "params.ckpt")
resnet.load_state_dict(torch.load("params.ckpt"))
