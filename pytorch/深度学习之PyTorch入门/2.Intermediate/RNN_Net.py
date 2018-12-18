#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-07 10:23:19
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms

# 配置设备
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
# 超参数设置
sequence_length = 28
input_size = 28
hidden_size = 128
num_classes = 10
num_layers = 2
batch_size = 100
num_epochs = 2
learning_rate = 0.01
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
# RNN model----多对一


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(device)  # 设置初始隐藏状态与输出状态
        c0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(device)

        # 输出的tensor shape是(batch_size, sequence_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        # 解码最后一个时间序列的隐藏状态
        out = self.fc(out[:, -1, :])
        return out


model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
# 定义Loss与优化算法
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        # 前向传播
        outputs = model(images)
        loss = loss_func(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print("Epoch[{}/{}], Step[{}/{}], Loss:{:.4f}".format(epoch +
                                                                  1, num_epochs, i + 1, total_step, loss.item()))

# 测试模型
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(
        100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'RNN_Net.ckpt')
