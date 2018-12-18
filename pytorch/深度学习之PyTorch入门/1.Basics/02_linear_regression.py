#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-06 16:11:27
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

# 超参数
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

# Toy dataset

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
# 线性回归模型
model = nn.Linear(input_size, output_size)
# Loss和优化算法
loss_func = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# 开始训练
for epoch in range(num_epochs):
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)
    # 前向传播
    output = model(inputs)
    loss = loss_func(output, targets)
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0:
        print("经过迭代%d次后，loss值是:%.2f" % (epoch + 1, loss.item()))

# 绘图
y_hat = model(torch.from_numpy(x_train)).detach().numpy()
# detach()方法是一个新的从当前图中分离的Variable
plt.plot(x_train, y_train, "ro", label="Original data")
plt.plot(x_train, y_hat, "bo", label="Fitting line")
plt.legend()
plt.show()
# 保存模型
torch.save(model.state_dict(), "linear_regression.ckpt")
