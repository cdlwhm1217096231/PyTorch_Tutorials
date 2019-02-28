#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-02-27 19:05:04
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

"""
1.使用DataParallel来使用多块GPU
2.使用如下方式把一个模型放到GPU上:
    device = torch.device("cuda:0")
    model.to(device)
3.使用DataParallel可以轻易的让模型并行运行在多个GPU上:
    model = nn.DataParallel(model)
"""

# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100

# 1.device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 2.制作虚拟数据集
class RandomDataSet(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


random_loader = DataLoader(dataset=RandomDataSet(
    input_size, data_size), batch_size=batch_size, shuffle=True)

# 3.简单模型
"""
简单模型，只接收一个输入，执行一个线性操作，然后得到结果。DataParallel能够用在任何模型上
"""


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("In Model: input size", input.size(),
              "output size: ", output.size())
        return output


# 4.创建一个模型和数据并行
"""
创建一个模型实例和检测是否有多个GPU。如果有多个GPU，使用nn.DataParallel来包装模型，最后通过model.to(device)把模型放到GPU上
"""
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)

# 5.运行模型
for data in random_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())
