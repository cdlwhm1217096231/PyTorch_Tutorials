#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-18 20:34:23
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

# 设定随机数种子
torch.manual_seed(2017)
# 从data.txt文件中读取数据
with open("./datasets/data.txt", "r") as f:
    data_list = []
    for line in f.readlines():
        line = line.strip().split(",")
        data_list.append(line)

# print(data_list)
data = [(float(i[0]), float(i[1]), float(i[2])) for i in data_list]
# 标准化
x0_max = max([i[0] for i in data])
x1_max = max([i[1] for i in data])
data = [(i[0] / x0_max, i[1] / x1_max, i[2]) for i in data]  # 预处理之后的data
x0 = list(filter(lambda x: x[-1] == 0.0, data))  # 第一类点
x1 = list(filter(lambda x: x[-1] == 1.0, data))   # 第二类点

x0_point = [i[0] for i in x0]
y0_point = [i[1] for i in x0]

x1_point = [i[0] for i in x1]
y1_point = [i[1] for i in x1]

plt.plot(x0_point, y0_point, "ro", label="x0_point")
plt.plot(x1_point, y1_point, "bo", label="x1_point")
plt.legend(loc='best')
plt.show()

# 将数据转化成numpy的类型,为转化成tensor数据类型做准备
np_data = np.array(data, dtype="float32")
x_data = torch.from_numpy(np_data[:, 0:2])  # tensor的shape是(100, 2)
y_label = torch.from_numpy(np_data[:, -1]).unsqueeze(1)  # tensor的shape是（100，1）


# 定义sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 画出sigmoid函数
x = np.arange(-10, 10.01, 0.01)
plt.plot(x, sigmoid(x), "r")
plt.title("sigmoid function")
plt.show()

# 将tensor转化成Variable
x_data = Variable(x_data)
y_label = Variable(y_label)

import torch.nn.functional as F


# 定义模型
W = Variable(torch.randn(2, 1), requires_grad=True)
b = Variable(torch.randn(1), requires_grad=True)


def logistic_regression(x):
    return torch.sigmoid(torch.mm(x, W) + b)


# 画出参数更新前的结果
W0 = W[0].item()
W1 = W[1].item()
b0 = b.item()
x = np.arange(0.2, 1, 0.01)
y = (-W0 * x - b0) / W1

plt.plot(x, y, "g", label="cutting line")
plt.plot(x0_point, y0_point, "ro", label="x0_point")
plt.plot(x1_point, y1_point, "bo", label="x1_point")
plt.legend(loc='best')
plt.title("decision boundary")
plt.show()


# 计算loss
def binary_loss(y_hat, y):
    logits = (y * y_hat.clamp(1e-12).log() + (1 - y)
              * (1 - y_hat).clamp(1e-12).log()).mean()
    return -logits


y_hat = logistic_regression(x_data)
loss = binary_loss(y_hat, y_label)
print(loss.item())

# 自动求导并更新参数
loss.backward()
W.data = W.data - 0.1 * W.grad.data
b.data = b.data - 0.1 * b.grad.data

# 算出一次更新之后的loss
y_pred = logistic_regression(x_data)
loss = binary_loss(y_pred, y_label)
print("一次更新之后的loss:", loss.item())


"""下面使用pytorch中的方法进行梯度下降更新参数"""
from torch import nn
W = nn.Parameter(torch.randn(2, 1))
b = nn.Parameter(torch.zeros(1))


def logistic_regression(x):
    return torch.sigmoid(torch.mm(x, W) + b)


optimizer = torch.optim.SGD([W, b], lr=1.0)

# 进行更新1000次
import time

start = time.time()
for e in range(1000):
    y_hat = logistic_regression(x_data)
    loss = binary_loss(y_hat, y_label)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 计算正确率
    mask = y_hat.ge(0.5).float()
    acc = (mask == y_label).sum().item() / y_label.shape[0]
    if (e + 1) % 200 == 0:
        print("Epoch:{}, Loss:{:.5f}, Acc:{:.5f}".format(e + 1, loss.item(), acc))
during = time.time() - start
print("所用时间:{:.3f}".format(during))
#  画出更新后的效果
W0 = W[0].item()
W1 = W[1].item()
b0 = b.item()

x = np.arange(0.2, 1, 0.01)
y = (-W0 * x - b0) / W1

plt.plot(x, y, "g", label="cutting line")
plt.plot(x0_point, y0_point, "ro", label="x0_point")
plt.plot(x1_point, y1_point, "bo", label="x1_point")
plt.legend(loc='best')
plt.title("decision boundary")
plt.show()


"""下面使用pytorch中的loss函数，不使用自己定义的binary_loss"""
loss_fun = nn.BCEWithLogitsLoss()
W = nn.Parameter(torch.randn(2, 1))
b = nn.Parameter(torch.randn(1))


def logistic_reg(x):
    return torch.mm(x, W) + b


optimizer = torch.optim.SGD([W, b], 1.)
y_hat = logistic_reg(x_data)
loss = loss_fun(y_hat, y_label)
print(loss.item())

start = time.time()
for e in range(1000):
    y_hat = logistic_regression(x_data)
    loss = binary_loss(y_hat, y_label)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 计算正确率
    mask = y_hat.ge(0.5).float()
    acc = (mask == y_label).sum().item() / y_label.shape[0]
    if (e + 1) % 200 == 0:
        print("Epoch:{}, Loss:{:.5f}, Acc:{:.5f}".format(e + 1, loss.item(), acc))
during = time.time() - start
print("所用时间:{:.3f}".format(during))


#  画出更新后的效果
W0 = W[0].item()
W1 = W[1].item()
b0 = b.item()

x = np.arange(0.2, 1, 0.01)
y = (-W0 * x - b0) / W1

plt.plot(x, y, "g", label="cutting line")
plt.plot(x0_point, y0_point, "ro", label="x0_point")
plt.plot(x1_point, y1_point, "bo", label="x1_point")
plt.legend(loc='best')
plt.title("decision boundary")
plt.show()
