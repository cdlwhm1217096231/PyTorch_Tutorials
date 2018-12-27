#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-26 22:34:10
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$


import torch
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as tfs


"""
RNN基本介绍：最简单的RNN有两种方式调用：torch.nn.RNNCell()和torch.nn.RNN()
区别:torch.nn.RNNCell()只接收序列中的单步的输入，且必须传入隐藏状态
RNN()可以接收一个序列的输入，默认会传入全0的隐藏状态，也可以自己声明隐藏状态传入
---------------------------------------------------------------------------------------
RNN()中的参数有：
input_size:输入x的特征维度
hidden_size:输出的特征的维度
num_layers:表示网络的层数
nonlinearity:表示选用的非线性激活函数,默认是'tanh'
bias:表示是否使用偏置，默认使用
batch_first:表示输入数据的形式，默认是False，即(seq, batch, feature),也就是说将序列长度放在第个一维度，batch放在第二个维度
dropout：表示是否在输出层应用dropout
bidirectional:表示是否使用双向RNN，默认是False

对于RNNCell(),里面的参数更少，只有inpu	t_size、hidden_size、bias、nonlinearity
"""

# 定义一个单步的RNN
rnn_single = nn.RNNCell(input_size=100, hidden_size=200)

# 访问其中的参数
print('RNNCell其中的参数:', rnn_single.weight_hh)

# 构造一个序列,长度seq=6,batch=5, feature=100
x = Variable(torch.randn(6, 5, 100))   # 这是rnn的输入格式
# 定义初始的隐藏状态
h_t = Variable(torch.zeros(5, 200))   # (batch, hidden_size)
# 传入rnn
out = []
for i in range(6):
    h_t = rnn_single(x[i], h_t)
    out.append(h_t)
print('RNNCell隐藏状态的值:', h_t)
print('RNNCell序列长度:', len(out))
print("RNNCell每个序列对应的输出维度:", out[0].shape)


"""下面是直接使用RNN"""
rnn_seq = nn.RNN(100, 200)
# 访问其中的参数
print('RNN第一层的参数:', rnn_seq.weight_hh_l0)
# 使用默认的全0隐藏状态
out, h_t = rnn_seq(x)
print('RNN最后时刻隐藏状态的值:', h_t)
print('RNN序列长度:', len(out))
# 不使用默认的全0隐藏状态,使用自己定义的初始隐藏状态
# (num_layers * num_direction, batch, hidden_size)
h_0 = Variable(torch.randn(1, 5, 200))
out, h_t = rnn_seq(x, h_0)
print("RNN最后时刻隐藏状态的值:", h_t)
print('RNN的输出:', out.shape)  # 输出的结果(seq, batch, feature)

"""
一般使用nn.RNN(),而不使用nn.RNNCell(),因为nn.RNN()可以避免手写循环，非常方便，一般情况下，不特别说明，默认使用全0来初始化隐藏状态
-------------------------------------------------------------------------------------------------

LSTM()和基本的RNN()一样，参数也相同，同时用nn.LSTM()和nn.LSTMCell()两种形式，情况与前面是相似的
"""
lstm_seq = nn.LSTM(50, 100, num_layers=2)  # 输入维度是100，输出维度是200,2层
print('LSTM第一层的权重:', lstm_seq.weight_hh_l0)

lstm_input = Variable(torch.randn(10, 3, 50))  # seq=10, batch=3, feature=50
out, (h, c) = lstm_seq(lstm_input)  # 全0状态初始化隐藏状态,c和h的两个隐藏状态的大小相同
print('LSTM的每个时刻隐藏状态h形状:', h.shape)
print('LSTM的每个时刻输出状态c形状:', c.shape)
print('LSTM的最终输出out形状:', out.shape)

# 不使用默认的全0隐藏状态,使用自己定义的初始隐藏状态
# (num_layers * num_direction, batch, hidden_size)
h_init = Variable(torch.randn(2, 3, 100))
c_init = Variable(torch.randn(2, 3, 100))
out, (h, c) = lstm_seq(lstm_input, (h_init, c_init))
print('LSTM的每个时刻隐藏状态h形状:', h.shape)
print('LSTM的每个时刻输出状态c形状:', c.shape)
print('LSTM的最终输出out形状:', out.shape)

"""
GRU单元介绍与前面两种情况类似
"""
gru_seq = nn.GRU(10, 20)  # 输入维度10，输出维度20
gru_input = Variable(torch.randn(3, 32, 10))  # seq=3, batch=32, feature=10
out, h = gru_seq(gru_input)
print('GRU单元的第一层权重:', gru_seq.weight_hh_l0)
print('GRU每个时刻隐藏状态h的形状:', h.shape)
print('GRU最终输出状态out的形状:', out.shape)
