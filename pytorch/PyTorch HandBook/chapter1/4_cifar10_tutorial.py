#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-02-26 19:09:02
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.autograd import Variable


"""
1.一般情况下，处理图像、文本、音频和视频数据时，可以使用标准的python包来加载数据到一个numpy数组中。然后把这个数组转换成torch.*tensor
    图像可以使用Pillow、OpenCV
    音频可以使用scipy、librosa
    文本可以使用原始python来加载，或者使用NLTK或spacy处理
2.针对图像任务，创建了一个包torchvision，它包含了处理一些基本图像数据集的方法。这些数据包括Imagenet、cifar10、mnist等。处理加载数据外，torchvision还包含图像转换器、torchvision.datasets和torch.utils.data.DataLoader
3.cifar10数据集说明:10个类别，图像的大小都是3*32*32的，即3通道的，32*32像素的。
"""
# 训练一个图像分类器
"""
1.使用torchvision加载和归一化cifar10训练集和测试集
2.定义一个卷积神经网络
3.定义损失和优化函数
4.在训练集上训练网络
5.在测试集上测试网络
"""

# 1.读取和归一化cifar10
"""
torchvision的输出是[0,1]的PIL图像，把它转化成范围是[-1,1]的tensor
"""
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
train_set = torchvision.datasets.CIFAR10(
    root="./dataset", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=4, shuffle=True)

test_set = torchvision.datasets.CIFAR10(
    root="./dataset", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=4, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 2.定义一个卷积神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)  # 两次使用的最大池化层格式是一样的2*2
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


cnn = CNN()

# 3.定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

# 4.训练网络
for epoch in range(20):  # 20次训练
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):  # i从0开始
        # 获取输入
        inputs, targets = data
        # 梯度置0
        optimizer.zero_grad()
        # 正向传播,反向传播,优化
        outputs = cnn(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        # 打印状态信息
        running_loss += loss.item()
        if i % 2000 == 1999:  # 每隔2000批次打印一次中间结果
            print("[%d, %5d] loss: %.3f" %
                  (epoch + 1, i + 1, running_loss / 2000))
            print("-" * 25)
print("完成训练！")
