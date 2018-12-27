#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-27 09:37:26
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


data_csv = pd.read_csv('./datasets/data.csv', usecols=[1])
plt.plot(data_csv)
plt.xlabel('时间序列', fontproperties='FangSong', fontsize=12)
plt.ylabel('10年的飞机月流量', fontproperties='FangSong', fontsize=12)
plt.title('原始数据集可视化', fontproperties='FangSong', fontsize=14)
plt.show()

# 数据预处理-----去除缺失值nan,数据标准化到0~1之间
data_csv = data_csv.dropna()
dataset = data_csv.values
dataset = dataset.astype('float32')
max_value = np.max(dataset)
min_value = np.min(dataset)
scalar = max_value - min_value
dataset = list(map(lambda x: x / scalar, dataset))

# 创建数据集,使用前面几个月的历史流量数据预测当月的流量数据，例如将前两个月的流量数据作为输入，当前月的流量数据作为输出，将数据分为训练集和测试集，将前几年的数据作为训练集，后两年的数据作为测试集

"""
	将整个数据集分为训练集(除后两年的数据外,其余数据都作为训练数据)和测试集(后两年的数据)
	X：前两个月的历史数据作为输入数据， Y：当前月的数据i+lookback作为预测数据
"""


def create_dataset(dataset, look_back=2):

    X, Y = [], []
    for i in range(len(dataset) - look_back):
        x = dataset[i:(i + look_back)]
        y = dataset[i + look_back]
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)


# 创建还输入和输出
X, Y = create_dataset(dataset)
# 划分训练数据和测试数据，70%作为训练数据
train_size = int(len(X) * 0.7)
test_size = len(X) - train_size
X = X[:train_size]
Y = Y[:train_size]
test_X = X[train_size:]
test_Y = Y[train_size:]

"""
由于RNN输入的数据是(seq, batch, feature),由于这里只有一个序列,所以batch=1,输入的feature是前两个月的数据，所以feature=2
"""
trainX = X.reshape(-1, 1, 2)
trainY = Y.reshape(-1, 1, 1)
test_X = test_X.reshape(-1, 1, 2)
train_torch_x = torch.from_numpy(trainX)
train_torch_y = torch.from_numpy(trainY)
test_torch_x = torch.from_numpy(test_X)


# 定义模型
class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(lstm_reg, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
        self.reg = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.rnn(x)    # (seq, batch, hidden)
        s, b, h = x.shape
        x = x.view(s * b, h)  # 转换成线性层的输入格式
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x


net = lstm_reg(input_size=2, hidden_size=4)
# 输入的维度是2,使用前两个月的数据预测当前月的数据,隐藏层的维度可以任意指定
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

# 开始训练
for epoch in range(1000):
    x = Variable(train_torch_x)
    y = Variable(train_torch_y)
    # 前向传播
    output = net(x)
    loss = loss_func(output, y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print("Epoch：{}, Loss:{:.5f}".format(epoch + 1, loss.item()))

# 测试模型
net = net.eval()
test_X = X.reshape(-1, 1, 2)
test_torch_X = torch.from_numpy(test_X)
test_variable_X = Variable(test_torch_X)
pred_variable_Y = net(test_variable_X)  # 测试集预测的结果

# 改变输出的格式
pred_np_Y = pred_variable_Y.view(-1).data.numpy()
# 画出实际预测的结果
plt.plot(pred_np_Y, 'r', label='prediction')
plt.plot(dataset, 'b', label='real')
plt.legend(loc='best')
plt.xlabel('时间序列', fontproperties='FangSong', fontsize=12)
plt.ylabel('10年的飞机月流量', fontproperties='FangSong', fontsize=12)
plt.title('预测结果可视化', fontproperties='FangSong', fontsize=14)
plt.show()
