#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-02-28 11:10:50
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

WINDOWS_SIZE = 2
raw_text = "We are about to study the idea of a computational process. Computational processes are abstract beings that inhabit computers. As they evolve, processes manipulate other abstract things called data. The evolution of a process is directed by a pattern of rules called a program. People create programs to direct processes. In effect, we conjure the spirits of the computer with our spells.".split(
    ' ')
word_list = list(set(raw_text))
word_dict = {w: i for i, w in enumerate(word_list)}
print("word_dict: \n", word_dict)
numbers_dict = {i: w for i, w in enumerate(word_list)}
print("numbers_dict: \n", numbers_dict)

# 方法1
cbow = []
for i in range(WINDOWS_SIZE, len(raw_text) - WINDOWS_SIZE):
    context_words = [word_dict[raw_text[i - 2]], word_dict[raw_text[i - 1]],  # inputs
                     word_dict[raw_text[i + 1]], word_dict[raw_text[i + 2]], ]  # target
    target_word = word_dict[raw_text[i]]
    cbow.append((context_words, target_word))
print("cbow: \n", cbow)

# 方法2
dataset = []
for i in range(WINDOWS_SIZE, len(raw_text) - WINDOWS_SIZE):
    context_words = [raw_text[i - 2], raw_text[i - 1],
                     raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    dataset.append((context_words, target))
print("dataset: \n", dataset)


# 建立模型
class CBOW(nn.Module):
    def __init__(self, n_word, n_dim, context_size):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(n_word, n_dim)
        self.linear1 = nn.Linear(2 * context_size * n_dim, 128)
        self.linear2 = nn.Linear(128, n_word)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(1, -1)
        x = self.linear1(x)
        x = F.relu(x, inplace=True)
        x = self.linear2(x)
        x = F.log_softmax(x)
        return x


model = CBOW(n_word=len(word_dict), n_dim=100, context_size=WINDOWS_SIZE)

# 可以使用GPU时
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)


for epoch in range(100):
    running_loss = 0

    for word_pairs in cbow:
        context_words, target_word = word_pairs
        context_words = Variable(torch.LongTensor(
            [context_words]))
        target_word = Variable(torch.LongTensor([target_word]))

        if torch.cuda.is_available():    # 如果在GPU上训练时
            context_words = context_words.cuda()
            target_word = target_word.cuda()
        # 前向传播，反向传播
        output = model(context_words)
        loss = criterion(output, target_word)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch:{}'.format(epoch + 1), "Loss:{:.4f}".format(
        running_loss / len(dataset)))

print("训练完成!")
