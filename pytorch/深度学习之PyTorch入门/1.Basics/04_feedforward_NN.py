#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-06 19:27:24
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST 数据集
train_dataset = torchvision.datasets.MNIST(root='./datasets/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./datasets/',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# 全连接神经网络只有一个隐藏层


class NeuralNet(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(NeuralNet, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    z1 = self.fc1(x)
    a1 = self.relu(z1)
    z2 = self.fc2(a1)
    return z2


model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# 定义Loss与优化算法
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
  for i, (images, labels) in enumerate(train_loader):
    # 将tensor放到已经配置好的设备上
    images = images.reshape(-1, 28 * 28).to(device)
    labels = labels.to(device)

    # 前向传播
    outputs = model(images)
    loss = loss_func(outputs, labels)

    # 反向传播与优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % 100 == 0:
      print("Epoch[{}/{}], Step[{}/{}], Loss:{:.4f}".format(epoch +
                                                            1, num_epochs, i + 1, total_step, loss.item()))

# 测试模型----测试时，不需要计算梯度
with torch.no_grad():
  correct = 0
  total = 0
  for images, labels in test_loader:
    images = images.reshape(-1, 28 * 28).to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

  print("在10000张测试图片上，神经网络的精度是:{}%".format(100 * correct / total))

# 保存模型
torch.save(model.state_dict(), "feedforward_NN.ckpt")
