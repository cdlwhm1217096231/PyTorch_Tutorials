#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-18 16:42:07
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 数据集
train_set = mnist.MNIST(root="./datasets/", train=True, download=True)
test_set = mnist.MNIST(root="./datasets/", train=False, download=True)
# 查看当中的一个数据
a_data, a_label = train_set[0]
print("此处的数据格式是PIL库中的格式:", a_data)
print(a_label)
a_data = np.array(a_data, dtype="float32")
print(a_data.shape)
print("numpy格式的数据:", a_data)


# 数据预处理
def data_tf(x):
    x = np.array(x, dtype="float32") / 255
    x = (x - 0.5) / 0.5  # 标准化
    x = x.reshape((-1,))  # flatten
    x = torch.from_numpy(x)
    return x  # 此处的x是tensor


# 重新载入数据，并声明数据的处理方法
train_set = mnist.MNIST(root="./datasets/", train=True,
                        transform=data_tf, download=True)
test_set = mnist.MNIST(root="./datasets/", train=False,
                       transform=data_tf, download=True)

a, a_label = train_set[0]
print("数据预处理后的结果:", a.shape)
print(a_label.data.item())

# 使用pytorch自带的dataloader定义一个数据迭代器
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)
# 上面这样做的目的是:使用数据迭代器，如果数据量大的话，无法将数据一次性读入内存，所以使用的是python迭代器，每次生成一个批次的数据
a, a_label = next(iter(train_data))
# 打印一个批次的数据大小
print(a.shape)
print(a_label.shape)
# 定义4层神经网络模型
net = nn.Sequential(
    nn.Linear(784, 400),
    nn.ReLU(),
    nn.Linear(400, 200),
    nn.ReLU(),
    nn.Linear(200, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
)
print(net)
#  定义loss损失函数
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), 1e-1)
# 开始训练
losses = []
acces = []
eval_losses = []
eval_acces = []
for e in range(20):   # 训练20次
    train_loss = 0
    train_acc = 0
    net.train()  # 训练模式
    for im, label in train_data:
        im = Variable(im)
        label = Variable(label)
        # 前向传播
        out = net(im)
        loss = loss_fn(out, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        train_acc += acc
    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))
    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    net.eval()  # 测试模式
    for im, label in test_data:
        im = Variable(im)
        label = Variable(label)
        out = net(im)
        loss = loss_fn(out, label)
        # 记录误差
        eval_loss += loss.item()
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        eval_acc += acc
    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))
    print("epch:{}, Train Loss:{:.6f}, Train Acc:{:.6f}, Eval Loss:{:.6f}, Eval Acc:{:.6f}".format(
        e + 1, train_loss / len(train_data), train_acc / len(train_data), eval_loss / len(test_data), eval_acc / len(test_data)))


# 画出loss和准确率曲线
plt.title("train loss")
plt.plot(np.arange(len(losses)), losses)
plt.show()


plt.plot(np.arange(len(acces)), acces)
plt.title("train acc")
plt.show()

plt.plot(np.arange(len(eval_losses)), eval_losses)
plt.title("test loss")
plt.show()


plt.plot(np.arange(len(eval_acces)), eval_acces)
plt.title("test acc")
plt.show()
