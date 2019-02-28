#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-02-28 15:47:20
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


torch.manual_seed(123)
"""
1.深度学习的实质是反向传播求导数，在pytorch中的autograd模块则实现了这个功能。在Tensor上的所有操作，autograd都能为它自动计算微分，避免手动计算微分。
2.从0.4版本开始，Variable正式合并入Tensor，Variable本来实现的自动微分功能，Tensor就能支持。还是可以使用
Variable(Tensor)，但这个操作其实什么都没做。
3.要使得Tensor有自动求导的功能，需要将属性tensor.requires_grad=True
"""


# 在创建张量时，通过设置requires_grad=True来告诉Pytorch需要对该张量进行自动的求导，pytorch会记录该张量的每一步操作历史并自动计算。

x = torch.randn(5, 5, requires_grad=True)
print("x: \n", x)

y = torch.randn(5, 5, requires_grad=True)
print("y:\n", y)

z = torch.sum(x + y)
print("z: \n", z)

# 简单的自动求导
z.backward()
print("x和y的梯度是: \n", x.grad, "\n", y.grad)

# 复杂的求导
z = x ** 2 + y**3
print(z)
# 返回值不是一个scalar，所以需要输入一个大小相同的张量作为参数，这里我们用ones_like函数根据x生成一个张量
z.backward(torch.ones_like(x))
print(x.grad, '\n', y.grad)

# 使用with torch.no_grad():禁止已经设置requires_grad=True的向量进行自动求导，这个方法在测试集上测试准确率的时候回经常用到!!!!
with torch.no_grad():
    print((x + y * 2).requires_grad)
