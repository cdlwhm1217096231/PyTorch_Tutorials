#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-02-26 14:15:49
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

"""
1.Basic Embedding Model
    1-1. NNLM(Neural Network Language Model)----> predict next word
"""

dtype = torch.FloatTensor
sentences = ["i like dog", "i love coffee", "i hate milk"]

word_list = " ".join(sentences).split()  # 制作词汇表
print(word_list)
word_list = list(set(word_list))  # 去除词汇表中的重复元素
print("去重后的word_list:", word_list)
word_dict = {w: i for i, w in enumerate(word_list)}  # 将每个单词对应于相应的索引
number_dict = {i: w for i, w in enumerate(word_list)}  # 将每个索引对应于相应的单词
n_class = len(word_dict)  # 单词的总数

# NNLM parameters
n_step = 2   # 根据前两个单词预测第3个单词
n_hidden = 2  # 隐藏层神经元的个数
m = 2  # 词向量的维度


# 由于pytorch中输入的数据是以batch小批量进行输入的，下面的函数就是将原始数据以一个batch为基本单位喂给模型
def make_batch(sentences):
    input_batch = []
    target_batch = []
    for sentence in sentences:
        word = sentence.split()
        input = [word_dict[w] for w in word[:-1]]
        target = word_dict[word[-1]]
        input_batch.append(input)
        target_batch.append(target)
    return input_batch, target_batch

# Model


class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        self.C = nn.Embedding(n_class, embedding_dim=m)
        self.H = nn.Parameter(torch.randn(n_step * m, n_hidden).type(dtype))
        self.W = nn.Parameter(torch.randn(n_step * m, n_class).type(dtype))
        self.d = nn.Parameter(torch.randn(n_hidden).type(dtype))
        self.U = nn.Parameter(torch.randn(n_hidden, n_class).type(dtype))
        self.b = nn.Parameter(torch.randn(n_class).type(dtype))

    def forward(self, x):
        x = self.C(x)
        x = x.view(-1, n_step * m)
        # x: [batch_size, n_step*n_class]
        tanh = torch.tanh(self.d + torch.mm(x, self.H))
        # tanh: [batch_size, n_hidden]
        output = self.b + torch.mm(x, self.W) + torch.mm(tanh, self.U)
        # output: [batch_size, n_class]
        return output


model = NNLM()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 制作输入
input_batch, target_batch = make_batch(sentences)
input_batch = Variable(torch.LongTensor(input_batch))
target_batch = Variable(torch.LongTensor(target_batch))


# 开始训练
for epoch in range(5000):
    optimizer.zero_grad()
    output = model(input_batch)
# output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
    loss = criterion(output, target_batch)
    if (epoch + 1) % 1000 == 0:
        print("Epoch:{}".format(epoch + 1), "Loss:{:.3f}".format(loss))
    loss.backward()
    optimizer.step()

# 预测
predict = model(input_batch).data.max(
    1, keepdim=True)[1]  # [batch_size, n_class]
print("predict: \n", predict)
# 测试
print([sentence.split()[:2] for sentence in sentences], "---->",
      [number_dict[n.item()] for n in predict.squeeze()])
