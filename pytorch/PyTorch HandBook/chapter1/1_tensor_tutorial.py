#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-02-25 10:38:15
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

from __future__ import print_function
import torch
import numpy as np

torch.manual_seed(123)
"""
pytorch是什么：
    1.作为numpy的替代者，可以使用gpu的强大计算能力
    2.提供最大的灵活性和高速的深度学习研究平台
"""

# 1.Tensor(张量)：与numpy中的ndarrays类似，但是在pytorch中Tensors可以使用GPU进行计算

# 创建一个5*3的矩阵，但未初始化
x = torch.empty(5, 3)
print("empty x: \n", x)

# 创建一个随机初始化的矩阵
x = torch.rand(5, 3)
print("random x: \n", x)

# 创建一个0填充的矩阵，数据类型为long
x = torch.zeros(5, 3, dtype=torch.long)
print("long Tensor: \n", x)

# 创建tensor并使用现有数据进行初始化
x = torch.tensor([5.5, 3])
print("initialize x way1: \n", x)

# 根据现有的张量创建新的张量，这些方法将重用输入张量的属性，例如：dtype，除非设置新的值进行覆盖
x = x.new_ones(5, 3, dtype=torch.double)  # new_*方法创建新的对象
print("new object tensor: \n", x)
x = torch.rand_like(x, dtype=torch.float)  # 覆盖dtype
print("x: \n", x)  # 对象的size是相同的，只是值和类型发生变化

# 获取size
"""
使用size方法与numpy中的shape属性返回的相同，张量也支持shape属性
"""
print("x size is:", x.size())  # 注意：torch.Size返回值是tuple类型，所以它支持tuple类型的所有操作


# 2.操作

# 加法1
y = torch.rand(5, 3)
print("x + y: \n", x + y)

# 加法2
print("x + y: \n", torch.add(x, y))
# 提供输出tensor作为参数
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print("result = \n", result)

# 替换in place
y.add_(x)  # add x to y
print("y: \n", y)
# 注：任何以_结尾的操作都会用结果替换原来的变量，例如x.copy_(y), x.t_(),都会改变x

# 可以使用与numpy索引方式相同的操作来进行对张量的操作
print("tensor indices: \n", x[:, 1])

# torch.view:可以改变张量的大小和维度，与numpy中的reshape类似
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # size中的-1表示从其他的维度进行判断
print("x size is: \n", x.size())
print("y size is: \n", y.size())
print("z size is: \n", z.size())

# 如果只有一个元素，可以使用item()方法来得到python数据类型的值
x = torch.randn(1)
print(x)
print(x.item())

# convert a torch tensor to numpy ndarray
a = torch.ones(5)
print("tensor a: \n", a)
b = a.numpy()
print("numpy b: \n", b)
a.add_(1)
print("tensor a:\n", a)
print("numpy b:\n", b)

# convert numpy data to torch tensor
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print("numpy a: \n", a)
print("tensor b: \n", b)
# 所有的 Tensor 类型默认都是基于CPU， CharTensor 类型不支持到 NumPy 的转换

# CUDA张量---使用to()方法可以将tensor移动到任何设备上
"""
is_available函数判断是否有cuda可以使用
torch.device：将张量移动到指定的设备上
"""
if torch.cuda.is_available():
    device = torch.device("cuda")  # 一个cuda设备对象
    y = torch.ones_like(x, device=device)  # 直接从GPU创建张量
    x = x.to(device)   # 或者直接使用.to("cuda")将张量移动到cuda中
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))  # .to()也会对变量的类型做更改
