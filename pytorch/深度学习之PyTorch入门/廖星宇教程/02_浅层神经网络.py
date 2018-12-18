#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Version: python 3.5.2
# Tools: Pycharm 2017.2.2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# 绘制决策面
def plot_decision_boundary(model, x, y):
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z = model(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.Spectral)
    plt.ylabel("x2")
    plt.xlabel("x1")
    plt.scatter(x[:, 0], x[:, 1], c=y.reshape(-1), s=40, cmap=plt.cm.Spectral)

np.random.seed(1)
m = 400  # 样本数量
N = int(m/2)  # 每一类的点的个数
D = 2  # 维度
x = np.zeros((m, D))
y = np.zeros((m, 1), dtype='uint8')  # label 向量，0 表示红色，1 表示蓝色
a = 4

for j in range(2):
    ix = range(N*j, N*(j+1))
    t = np.linspace(j*3.12, (j+1)*3.12, N) + np.random.randn(N)*0.2 # theta
    r = a*np.sin(4*t) + np.random.randn(N)*0.2  # radius
    x[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j

plt.scatter(x[:, 0], x[:, 1], c=y.reshape(-1), s=40, cmap=plt.cm.Spectral)
plt.title("dataset visualize")
plt.show()
"""先使用逻辑回归解决分类问题"""
x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()
W = nn.Parameter(torch.randn(2, 1))
b = nn.Parameter(torch.zeros(1))
optimizer = optim.SGD([W, b], lr=1e-1)  # 定义优化器
loss_func = nn.BCEWithLogitsLoss()   # 定义损失函数


# 定义逻辑回归模型
def logistic_regression(x):
    s = torch.mm(x, W) + b   # 没有经过激活函数
    return s
# 开始训练
for epoch in range(100):
    y_pred = logistic_regression(Variable(x))
    loss = loss_func(y_pred, Variable(y))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 20 == 0:
        print("经过%d迭代后,Loss的值是:%.2f" % (epoch+1, loss.item()))


# 经过激活函数后的模型
def plot_logistic(x):
    x = Variable(torch.from_numpy(x).float())
    out = torch.sigmoid(logistic_regression(x))
    out = (out > 0.5) * 1
    return out.data.numpy()

plot_decision_boundary(lambda x: plot_logistic(x), x.numpy(), y.numpy())
plt.title("logistic Regression")
plt.show()
"""下面使用神经网络构建分类器"""
# 定义网络参数
W1 = nn.Parameter(torch.randn(2, 4) * 0.01)  # 输入层2个神经元，隐藏层4个神经元
b1 = nn.Parameter(torch.zeros(4))
W2 = nn.Parameter(torch.randn(4, 1) * 0.01)  # 输出层一个神经元
b2 = nn.Parameter(torch.zeros(1))


# 定义模型
def two_network(x):
    z1 = torch.mm(x, W1) + b1   # 注意此处使用的是z1 = x*w + b与吴恩达视频表示的方法不同!
    a1 = torch.tanh(z1)  # 经过激活函数之后的值
    z2 = torch.mm(a1, W2) + b2
    return z2
# 定义优化算法和损失函数
optimizer = optim.SGD([W1, W2, b1, b2], 1.)
loss_func = nn.BCEWithLogitsLoss()
# 开始训练
for epoch in range(10000):
    out = two_network(Variable(x))
    loss = loss_func(out, Variable(y))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 1000 == 0:
        print("经过%d迭代后,Loss的值是:%.2f" % (epoch+1, loss.item()))


# 输出层经过激活函数后，预测分类
def plot_network(x):
    x = Variable(torch.from_numpy(x).float())
    z1 = torch.mm(x, W1) + b1
    a1 = torch.tanh(z1)
    z2 = torch.mm(a1, W2) + b2
    out = torch.sigmoid(z2)
    out = (out > 0.5) * 1
    return out.data.numpy()
# 绘制出决策面
plot_decision_boundary(lambda x: plot_network(x), x.numpy(), y.numpy())
plt.title("two layer network")
plt.show()




"""下面介绍Sequential与Module"""
# Sequential 允许我们构建序列化的模块，而 Module 是一种更加灵活的模型定义方式
# Sequential方法构建模型
seq_net = nn.Sequential(
    nn.Linear(2, 4),  # 输入层是2个神经元，隐藏层是4个神经元   xw + b
    nn.Tanh(),
    nn.Linear(4, 1)  # 隐藏层是4个神经元，输出层是1个神经元
)
# 序列模块可以通过索引访问每一层
print(seq_net[0])
# 打印出第一层的权重
print(seq_net[0].weight)
# 通过parameters可以获得模型的参数
param = seq_net.parameters()
# 定义优化算法
optimizer = optim.SGD(param, 1.)
# 训练10000次
for epoch in range(10000):
    out = seq_net(Variable(x))
    loss = loss_func(out, Variable(y))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 1000 == 0:
        print("经过%d次迭代后，Loss的值是:%.2f" % (epoch+1, loss.item()))


# 模型预测开始
def pplot_seq(x):
    out = torch.sigmoid(seq_net(Variable(torch.from_numpy(x).float()))).data.numpy()
    out = (out > 0.5) * 1
    return out
# 绘制决策面
plot_decision_boundary(lambda x: pplot_seq(x), x.numpy(), y.numpy())
plt.title("sequential")
plt.show()
"""下面是模型的保存与加载"""
# 模型的保存方法1
torch.save(seq_net, "save_seq_net.pth")
# 模型的加载方法1
seq_net1 = torch.load("save_seq_net.pth")
print(seq_net1)
print(seq_net1[0].weight)
# 模型的保存方法2----推荐方法2
torch.save(seq_net.state_dict(), "save_seq_net_params.pth")   # 只保存参数不保存模型
# 模型的加载方法2
seq_net2 = nn.Sequential(
    nn.Linear(2, 4),
    nn.Tanh(),
    nn.Linear(4, 1)
)
seq_net2.load_state_dict(torch.load("save_seq_net_params.pth"))
print(seq_net2)
print(seq_net2[0].weight)
"""
下面使用Module进行自定义模型           # 很重要！
class 网络名字(nn.Module):
    def __init__(self, 一些定义的参数):
        super(网络名字, self).__init__()
        self.layer1 = nn.Linear(num_input, num_hidden)
        self.layer2 = nn.Sequential(...)
        ...

        定义需要用的网络层

    def forward(self, x):   # 定义前向传播
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x = x1 + x2
        ...
        return x

"""


class Module_model(nn.Module):
    def __init__(self, num_input, num_hidden, num_output):
        super(Module_model, self).__init__()
        self.layer1 = nn.Linear(num_input, num_hidden)
        self.layer2 = nn.Tanh()
        self.layer3 = nn.Linear(num_hidden, num_output)

    def forward(self, x):
        z1 = self.layer1(x)
        a1 = self.layer2(z1)
        z2 = self.layer3(a1)
        return z2

mo_net = Module_model(2, 4, 1)
# 访问模型中的某层可以直接通过名字
# 第一层
l1 = mo_net.layer1
print(l1)
# 打印第一层的权重
print(l1.weight)
optimizer = optim.SGD(mo_net.parameters(), 1.)
for epoch in range(10000):
    out = mo_net(Variable(x))
    loss = loss_func(out, Variable(y))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 1000 == 0:
        print("经过%d次迭代后，Loss的值是:%.2f" % (epoch + 1, loss.item()))

# 保存模型
torch.save(mo_net.state_dict(), 'module_net.pth')
"""练习"""
net = nn.Sequential(
    nn.Linear(2, 10),
    nn.Tanh(),
    nn.Linear(10, 10),
    nn.Tanh(),
    nn.Linear(10, 10),
    nn.Tanh(),
    nn.Linear(10, 1)
)
optimizer = optim.SGD(net.parameters(), lr=1.)
for epoch in range(20000):
    out = net(Variable(x))
    loss = loss_func(out, Variable(y))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 1000 ==0:
        print("经过%d次迭代后，Loss的值是:%.2f" % (epoch + 1, loss.item()))


def plot_net(x):
    out = torch.sigmoid(net(Variable(torch.from_numpy(x).float()))).data.numpy()
    out = (out > 0.5) * 1
    return out

plot_decision_boundary(lambda x: plot_net(x), x.numpy(), y.numpy())
plt.title('sequential2')
plt.show()
