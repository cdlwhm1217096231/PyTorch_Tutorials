#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-10 10:59:22
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
import torch.utils.data.dataloader as dataloader
import torch.nn.functional as F
import sys

# 数据预处理的方法
"""
中心化: 在每个特征的维度上减去对应的均值，得到0均值的特征
标准化: 在数据变成0均值后，使得不同的特征维度有着相同的规模，可以除以标准差近似为一个标准正态分布
或者可以根据最大值和最小值将其转化为-1到1之间
"""

# 批标准化
"""
对于深层的神经网络，网络的非线性层会使得输出的结果变得相关，不再满足一个标准的N(0,1)分布
批标准化就是对每层的网络输出，对其做一个归一化，使其服从标准的正态分布，能够较好的进行训练，加快收敛速度
"""
sys.path.append("..")


def simple_batch_norm_1d(x, gamma, beta):
    eps = 1e-5
    x_mean = torch.mean(x, dim=0, keepdim=True)  # 保留维度进行 broadcast
    x_var = torch.mean((x - x_mean) ** 2, dim=0, keepdim=True)
    x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
    return gamma.view_as(x_mean) * x_hat + beta.view_as(x_mean)


#  进行验证
x = torch.arange(15).view(5, 3).float()
gamma = torch.ones(x.shape[1])
beta = torch.zeros(x.shape[1])
print('before BN: ')
print(x)
y = simple_batch_norm_1d(x, gamma, beta)
print('after BN: ')
print(y)
# 以下能够区分训练状态和测试状态的批标准化方法


def batch_norm_1d(x, gamma, beta, is_training, moving_mean, moving_var, moving_momentum=0.1):
    eps = 1e-5
    x_mean = torch.mean(x, dim=0, keepdim=True)  # 保留维度进行 broadcast
    x_var = torch.mean((x - x_mean) ** 2, dim=0, keepdim=True)
    if is_training:
        x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
        moving_mean[:] = moving_momentum * \
            moving_mean + (1. - moving_momentum) * x_mean
        moving_var[:] = moving_momentum * \
            moving_var + (1. - moving_momentum) * x_var
    else:
        x_hat = (x - moving_mean) / torch.sqrt(moving_var + eps)
    return gamma.view_as(x_mean) * x_hat + beta.view_as(x_mean)
