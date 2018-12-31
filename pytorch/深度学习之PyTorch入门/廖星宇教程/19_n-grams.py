#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-31 10:23:56
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231
# @Version : $Id$


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


"""
使用词嵌入来训练语言模型n-grams:根据一句话的前面几个词，预测后面几个词
给定一个句子T，这个句子由w1 w2 w3 等n个单词组成p(T) = p(w1)p(w2|w1)p(w3|w2w1)....p(wn|wn-1*Wn-2*...*w2*w1)
简化上述模型,对于一个词，并不需要前面所有的词作为条件概率，即一个词可能只与前面的几个词有关，这就是马尔科夫假设

这里的条件概率的计算方法：
    1.传统方法：统计语料库中的每个单词的出现频率，然后使用贝叶斯定理来估计这个条件概率
    2.改进方法: 使用词嵌入代替每个单词，然后使用RNN进行条件概率的计算，再最大化这个条件概率。这样不仅修改词嵌入，同时
    能够使模型可以根据计算的条件概率对其中的一个单词进行预测。
"""

################### 具体的实现 ####################
CONTEXT_SIZE = 2   # 依据的单词的数量，即根据前面2个单词来预测这个单词
EMBEDDING_DIM = 10  # 词向量的维度
# 语料库
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

# 建立训练集
trigram = [((test_sentence[i], test_sentence[i + 1]), test_sentence[i + 2])
           for i in range(len(test_sentence) - 2)]
print("总的数据数量:", len(trigram))
# 打印第一个数据
print(trigram[0])

# 建立每个词与数字的编码，据此构建词嵌入
vocb = set(test_sentence)   # 使用set将重复的元素去掉
word_to_idx = {word: i for i, word in enumerate(vocb)}   # 单词到索引的映射
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}  # 索引到单词的映射
print(word_to_idx)
# 定义模型


class n_grams(nn.Module):
    def __init__(self, vocab_size, context_size=CONTEXT_SIZE, n_dim=EMBEDDING_DIM):
        super(n_grams, self).__init__()

        self.embed = nn.Embedding(vocab_size, n_dim)
        self.classify = nn.Sequential(
            nn.Linear(context_size * n_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, vocab_size),
        )

    def forward(self, x):
        voc_embed = self.embed(x)  # 得到词嵌入
        voc_embed = voc_embed.view(1, -1)  # 将两个词向量拼接在一起
        out = self.classify(voc_embed)
        return out


# 最后输出的是一个条件概率，使用交叉熵来定义loss
net = n_grams(len(word_to_idx))

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, weight_decay=1e-5)

for e in range(100):
    train_loss = 0
    for word, label in trigram:
        word = Variable(torch.LongTensor(
            [word_to_idx[i] for i in word]))  # 将两个词作为输入
        label = Variable(torch.LongTensor([word_to_idx[label]]))
        # 前向传播
        out = net(word)
        loss = loss_func(out, label)
        train_loss += loss.item()
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (e + 1) % 20 == 0:
        print("Epoch:{}, Loss:{:.6f}".format(e + 1, train_loss / len(trigram)))


# 测试模型
net = net.eval()
# 测试以下结果1
word, label = trigram[19]
print("input:{}".format(word))
print("label:{}".format(label))
print()
word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
out = net(word)
pred_label_idx = out.max(1)[1].item()
pred_word = idx_to_word[pred_label_idx]
print("real word is {}, predicted word is {}".format(label, pred_word))

# 测试以下结果2
word, label = trigram[75]
print("input:{}".format(word))
print("label:{}".format(label))
print()
word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
out = net(word)
pred_label_idx = out.max(1)[1].item()
pred_word = idx_to_word[pred_label_idx]
print("real word is {}, predicted word is {}".format(label, pred_word))
