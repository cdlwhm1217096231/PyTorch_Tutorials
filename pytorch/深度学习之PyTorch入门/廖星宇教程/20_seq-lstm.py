#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-31 10:24:46
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231
# @Version : $Id$

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

"""
LSTM做词性预测
# 整体思路：
首先,将一个单词看作是一个序列，对每个字符构建词嵌入，然后输入lstm，只取最后一个输出作为预测结果，整个单词的字符串能够
形成一种记忆的特性，有助于更好的预测词性。
接着，将这个单词与前面几个单词构成一个序列，对这些单词构成的新的词嵌入，最后输出的结果是单词的词性，即根据前面几个词
的信息对这个词的词性进行分类。
"""

training_data = [("The dog ate the apple".split(),
                  ["DET", "NN", "V", "DET", "NN"]),
                 ("Everybody read that book".split(),
                  ["NN", "V", "DET", "NN"])]
# 对单词和标签进行编码
word_to_idx = {}
tag_to_idx = {}
for content, tag in training_data:
    for word in content:
        if word.lower() not in word_to_idx:
            word_to_idx[word.lower()] = len(word_to_idx)
    for label in tag:
        if label.lower() not in tag_to_idx:
            tag_to_idx[label.lower()] = len(tag_to_idx)
print(word_to_idx)
print(tag_to_idx)
# 然后对字母进行编码
alphabet = "abcdefghijklmnopqrstuvwxyz"
char_to_idx = {}
for i in range(len(alphabet)):
    char_to_idx[alphabet[i]] = i
print(char_to_idx)


# 构建训练数据
def make_sequence(x, dic):   # 字符编码
    # x:输入的序列数据（一个单词，或者由字符串组成的列表）
    # dic:字典规定了编码的方式
    idx = [dic[i.lower()] for i in x]
    idx = torch.LongTensor(idx)
    return idx


# 测试上面的函数，输入是一个单词，单词中的每个字符组成一个序列，dic是char_to_idx是将字符映射成索引的字典
result = make_sequence('apple', char_to_idx)
print(result)
# 测试上面的函数，输入的是一句话，每句话中的每个单词组成一个序列，dic是将每个单词映射成对应的索引
print(training_data[1][0])
print(make_sequence(training_data[1][0], word_to_idx))


# 构建单个字符的LSTM模型
class char_lstm(nn.Module):
    def __init__(self, n_char, char_dim, char_hidden):
        super(char_lstm, self).__init__()

        self.char_embed = nn.Embedding(n_char, char_dim)  # 构建单个字符的词嵌入层
        # 构建单个字符的lstm,输入的维度char_dim, 输出的维度char_hidden
        self.lstm = nn.LSTM(char_dim, char_hidden)

    def forward(self, x):
        x = self.char_embed(x)   # 前向传播，将单个字符输入到单个字符的词嵌入层
        out, _ = self.lstm(x)  # 单个单词的输出结果(seq, batch, hidden)
        return out[-1]   # (batch, hidden)


# 构建词性分类的LSTM模型
class lstm_tagger(nn.Module):
    def __init__(self, n_word, n_char, char_dim, word_dim, char_hidden, word_hidden, n_tag):
        super(lstm_tagger, self).__init__()

        self.word_embed = nn.Embedding(n_word, word_dim)  # 构建每个单词的词嵌入层
        self.char_lstm = char_lstm(
            n_char, char_dim, char_hidden)   # 字符级的LSTM模型输出的结果
        self.word_lstm = nn.LSTM(
            word_dim + char_hidden, word_hidden)  # 构建词性分类的LSTM模型
        self.classify = nn.Linear(word_hidden, n_tag)  # 词性的分类结果

    def forward(self, x, word):
        char = []
        for w in word:  # 遍历每个单词中的字符,对每个字符做lstm
            char_list = make_sequence(w, char_to_idx)
            # unsqueeze函数对数据进行维度扩充;squeeze()函数对数据维度进行压缩
            # （seq, batch, feature)满足lstm的输入
            char_list = char_list.unsqueeze(1)
            char_infor = self.char_lstm(
                Variable(char_list))  # (batch, char_hidden)
            char.append(char_infor)
        char = torch.stack(char, dim=0)   # (seq, batch, feature)

        x = self.word_embed(x)  # (batch, seq, word_dim)
        x = x.permute(1, 0, 2)  # 改变顺序
        x = torch.cat((x, char), dim=2)  # 沿着特征通道将每个词的词嵌入和字符 lstm 输出的结果拼接在一起
        x, _ = self.word_lstm(x)

        s, b, h = x.shape
        x = x.view(-1, h)  # 重新 reshape 进行分类线性层
        out = self.classify(x)
        return out


net = lstm_tagger(len(word_to_idx), len(char_to_idx),
                  10, 100, 50, 128, len(tag_to_idx))
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)

# 开始训练
for e in range(300):
    train_loss = 0
    for word, tag in training_data:
        word_list = make_sequence(
            word, word_to_idx).unsqueeze(0)  # 添加第一维 batch
        tag = make_sequence(tag, tag_to_idx)

        word_list = Variable(word_list)
        tag = Variable(tag)

        # 前向传播
        out = net(word_list, word)
        loss = loss_func(out, tag)
        train_loss += loss.item()
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (e + 1) % 50 == 0:
        print("Epoch:{}, Loss:{:.5f}".format(
            e + 1, train_loss / len(training_data)))

# 观测预测效果
net = net.eval()
test_sent = "Everybody ate the apple"
test = make_sequence(test_sent.split(), word_to_idx).unsqueeze(0)
out = net(Variable(test), test_sent.split())
print("out:", out)
print("tag_to_idx:", tag_to_idx)
