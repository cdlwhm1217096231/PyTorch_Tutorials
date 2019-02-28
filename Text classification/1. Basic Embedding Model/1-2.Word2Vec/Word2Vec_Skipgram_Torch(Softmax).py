#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-02-26 22:08:42
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt


"""
1.Basic Embedding Model
Word2Vec(Skip-gram)--->Embedding Words and Show Graph
"""

# a sentence is composed of three words.
dtype = torch.FloatTensor
sentences = ["i like dog", "i like cat", "i like animal",
             "i love Curry", "i like Durant", "i admire Harden",
             "i hate pig", "i like apple", "i hate banana",
             "i like read book", "i like watch film", "cat dog like"]

word_sequence = " ".join(sentences).split()
print("word_sequence:", word_sequence)


word_list = list(set(word_sequence))
print("word_list:", word_list)

word_dict = {w: i for i, w in enumerate(word_list)}
print("word_dict:", word_dict)


# Word2Vec Parameter
batch_size = 20
vocab_size = len(word_list)
embedding_size = 2


def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        random_inputs.append(np.eye(vocab_size)[data[i][0]])  # center word
        random_labels.append(data[i][1])  # context words
    return random_inputs, random_labels


# window size = 1
skip_grams = []
for i in range(1, len(word_sequence) - 1):
    center_word = word_dict[word_sequence[i]]
    context = [word_dict[word_sequence[i - 1]],
               word_dict[word_sequence[i + 1]]]
    for word in context:
        # (center_word,word)构成一对word pairs
        skip_grams.append([center_word, word])


# Model
class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()
        # input layer--->project layer,W是隐藏层的参数矩阵(vocab_size, embedding_size)
        self.W = nn.Parameter(-2 * torch.rand(vocab_size,
                                              embedding_size) + 1).type(dtype)
        # project layer----> input layer,WO是输出层的参数矩阵(embedding_size, vocab_size)
        self.WO = nn.Parameter(-2 * torch.rand(embedding_size,
                                               vocab_size) + 1).type(dtype)

    def forward(self, X):
        # X: [batch_size, vocab_size]
        # project layer: [batch_size, embedding_size]
        project_layer = torch.matmul(X, self.W)
        output_layer = torch.matmul(project_layer, self.WO)
        return output_layer


model = Word2Vec()
print("模型参数: \n", list(model.parameters()))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
for epoch in range(10000):
    input_batch, target_batch = random_batch(skip_grams, batch_size)
    input_batch = Variable(torch.Tensor(input_batch))
    target_batch = Variable(torch.LongTensor(target_batch))

    optimizer.zero_grad()
    output = model(input_batch)
    loss = criterion(output, target_batch)
    if (epoch + 1) % 1000 == 0:
        print("Epoch:{}".format(epoch + 1), "Loss:{:.4f}".format(loss.item()))
    loss.backward()
    optimizer.step()
print("训练完成!")


for i, label in enumerate(word_list):
    W, WO = model.parameters()
    x, y = float(W[i][0]), float(W[i][1])
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2),
                 textcoords='offset points', ha='right', va='bottom')
plt.show()
