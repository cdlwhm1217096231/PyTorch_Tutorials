#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-31 10:25:04
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231
# @Version : $Id$

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


"""
 词嵌入在NLP中的使用,如何表示一个单词，使用one-hot编码是不合理的，因为得到的向量是一个高维、稀疏的向量
 无法表示单词之间的相似性
 word-embedding：就是用一个向量来表示一个词语，但是这个词语不是随机的。需要将每个词汇都要有一个特定的向量去表示
 有一些词汇之间的词性是相似的，需要是用词汇去表示相似含义的词汇之间的相似性，使用余弦相似性来衡量
"""

# 使用pytorch实现词嵌入,只要调用torch.nn.Embedding(m,n),m:单词的总数,n:词嵌入的维度，
# 实际上词嵌入相当于是一个大矩阵，矩阵的每行表示一个单词


######################开始#########################

# 定义词嵌入
embeds = nn.Embedding(2, 5)   # 2个单词，维度是5
# 得到词嵌入矩阵
print(embeds.weight)

# 通过weight得到的整个词嵌入矩阵，这个词嵌入矩阵是一个可改变的参数。在网络训练中参数会不断更新，同时词嵌入的数值可以直接进行修改

# 直接手动修改词嵌入的值
embeds.weight.data = torch.ones(2, 5)
print(embeds.weight)

# 访问第50个词的词向量
embeds = nn.Embedding(100, 10)
 # 访问其中一个单词的词向量，直接调用定义好的词嵌入，必传入一个Variable，且类型是LongTensor
single_word_embed = embeds(Variable(torch.LongTensor([50])))  
print(single_word_embed, single_word_embed.shape)


# 词嵌入怎么得到的？使用skip-grams或者CBOW模型，在模型学习过程中，得到的额外的学习附加品
