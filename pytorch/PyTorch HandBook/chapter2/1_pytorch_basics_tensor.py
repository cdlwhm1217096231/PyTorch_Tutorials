#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-02-27 19:38:22
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

x = torch.randn(2, 3)
print("x: \n", x)
print("查看x的大小,方法1:", x.shape)
print("查看x的大小,方法2:", x.size())

# 生成多维张量
y = torch.rand(2, 3, 4, 5)
print("y: \n", y)
print("y的大小是:", y.size())
"""
0阶张量称为标量
1阶张量称为向量
2阶张量称为矩阵
3阶及以上称为多维张量
"""

scalar = torch.tensor(3.1415)
print("标量:", scalar)
print("标量的大小:", scalar.size())

# 对于标量，直接使用.item()方法从中取出对应的python对象的值
print(scalar.item())
# 当一个张量中只有一个元素时，也可以调用tensor.item()方法
tensor = torch.tensor(3.1415)
print("tensor:", tensor)
print("tensor.size()", tensor.size())
print("tensor.item()", tensor.item())

# 基本类型
"""
Tensor中的基本类型有5种：
    torch.FloatTensor(默认)、torch.LongTensor、torch.IntTensor、torch.ShortTensor、torch.DoubleTensor
"""
short_t = tensor.short()
print(short_t)

# numpy转换
a = torch.randn(3, 2)
# tensor转numpy
numpy_a = a.numpy()
print(numpy_a)
# numpy转tensor
np_data = np.arange(10)
tensor_data = torch.from_numpy(np_data)
print(tensor_data)

# 设备间转换
cpu_a = torch.rand(4, 3)
print(cpu_a.type())

# gpu_a = cpu_a.cuda() 使用.cuda()方法，将tensor从cpu移动到gpu
# print(gpu_a.type())

# cpu_b = gpu_a.cpu()  使用.cpu()方法，将tensor从gpu移动到cpu
# print(cpu_b.type())


# 如果存在多个GPU，使用to方法来确定使用哪个设备
# 使用torch.cuda.is_available()来判断是否有GPU设备
device = torch.device("cpu:0" if torch.cuda.is_available() else "cpu")
print(device)

# 将tensor传送到设备
# gpu_b = cpu_b.to(device)
# print(gpu_b.type())


# 初始化
rnd = torch.rand(5, 3)  # 服从0到1之间的均匀分布
print(rnd)

one = torch.ones(2, 3)  # 用1填充
print(one)

zeros = torch.zeros(3, 4)  # 用0填充
print(zeros)

# 初始化一个单位矩阵，即对角线为1 其他为0
eye = torch.eye(2, 2)
print(eye)

# 常用方法
x = torch.randn(3, 3)
print(x)

# 沿着行方向取最大值
max_value, max_idx = torch.max(x, dim=1)
print(max_value, max_idx)

# 每行 x 求和
sum_x = torch.sum(x, dim=1)
print(sum_x)

y = torch.randn(3, 3)
# add 完成后x的值改变了
x.add_(y)  # 以_为结尾的，均会改变调用值
print(x)
