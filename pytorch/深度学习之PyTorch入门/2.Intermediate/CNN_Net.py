#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-06 19:41:51
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 超参数设置
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# MNIST 数据集
train_dataset = torchvision.datasets.MNIST(
    root="./datasets/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(
    root="./datasets/", train=False, transform=transforms.ToTensor())

# Data Loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False)
# 定义模型


class CNN_Net(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(7 * 7 * 32, out_features=num_classes)

    def forward(self, x):
        z1 = self.layer1(x)
        a1 = self.layer2(z1)
        z2 = a1.reshape(a1.size(0), -1)   # 进入全连接层之前，需要将池化层输出的特征flatten
        a2 = self.fc(z2)
        return a2


model = CNN_Net(num_classes).to(device)
# 定义Loss与优化算法
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 开始训练
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = loss_func(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print("Epoch[{}/{}], Step[{}/{}], Loss:{:.4f}".format(epoch +
                                                                  1, num_epochs, i + 1, total_step, loss.item()))
# 测试模型
model.eval()  # 测试模式（batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print("在测试集上的精度:{}%".format(100 * correct / total))


#  保存模型
torch.save(model.state_dict(), "CNN_Net.model")
