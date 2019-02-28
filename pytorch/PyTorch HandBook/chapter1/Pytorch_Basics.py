#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-02-22 20:36:05
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import torch
import numpy as np
torch.manual_seed(1234)  # 设置随机数种子

print("Pytorch目前的版本:{}".format(torch.__version__))

# Tensor
"""
标量是一个单一数字
向量是一个数组
矩阵是一个二维数组
Tensors是一个n维数组
"""
# creating tensors


def describe(x):
    print("Type:{}".format(x.type()))
    print("Shape/Size:{}".format(x.shape))
    print("Value: \n{}".format(x))


describe(torch.Tensor(2, 3))
# random create tensor
x = torch.rand(2, 3)
describe(x)

describe(torch.zeros(2, 3))  # initialize tensors of zeros

x = torch.ones(2, 3)
describe(x)
x.fill_(5)  # use 5 fill in tensor x
describe(x)

# tensors can be initialize and filled in place
x = torch.Tensor(3, 4).fill_(5)
print(x.type())
print(x.shape)
print(x)

# tensor can be initialized from a list
x = torch.Tensor([[1, 2], [2, 4]])
describe(x)

# tensor can be initilized from numpy matrix
npy = np.random.rand(2, 3)
describe(torch.from_numpy(npy))
print("numpy中的数据类型：{}".format(npy.dtype))

# FloatTensor has been the default tensor, can be replaced all along
x = torch.arange(6).view(2, 3)
describe(x)

x = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
describe(x)
x = x.long()  # convert to long tensor
describe(x)

# assign tensor type when create a tensor
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
describe(x)
x = x.float()  # convert to float tensor
describe(x)

# create random tensor
x = torch.randn(2, 3)
describe(x)

describe(torch.add(x, x))  # add a tensor to another tensor
describe(x + x)  # equal to x+x

x = torch.arange(6)  # 1D tensor
describe(x)

x = x.view(2, 3)  # reshape 1D tensor to 2D tensor
describe(x)

# sum tensor
describe(torch.sum(x, dim=0))  # row direacton
describe(torch.sum(x, dim=1))  # columns direaction

# transpose tensor
describe(torch.transpose(x, 0, 1))

# slice tensor
x = torch.arange(6).view(2, 3)
describe(x)
describe(x[:1, :2])
describe(x[0, 1])

# index tensor
indices = torch.LongTensor([0, 2])  # columns index
describe(torch.index_select(x, dim=1, index=indices))  # columns select

indices = torch.LongTensor([0, 0])  # rows index
describe(torch.index_select(x, dim=0, index=indices))  # rows select

row_indices = torch.arange(2).long()
col_indices = torch.LongTensor([0, 1])
describe(x[row_indices, col_indices])

#  convert tensor to numpy data
x = torch.LongTensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
describe(x)
print(x.dtype)
print(x.numpy().dtype)  # tensor convert to numpy

# convert a float tensor to longtensor
x = torch.FloatTensor([1, 2, 3])
x = x.long()
describe(x)

# create a vector of incremental numbers
x = torch.arange(0, 10)
print(x)

# Sometimes it's useful to have an integer-based arange for indexing
x = torch.arange(0, 10).long()
print(x)

# reshape tensor
x = torch.arange(0, 20)
print(x.view(1, 20))
print(x.view(2, 10))
print(x.view(4, 5))
print(x.view(10, 2))

# add size 1 dimensions,which can be useful for combining with other tensors,also is called broadcasting
x = torch.arange(12).view(3, 4)
y = torch.arange(4).view(1, 4)
z = torch.arange(3).view(3, 1)
print(x)
print(y)
print(z)
print(x + y)
print(x + z)

# unsqueeze and squeeze will add and remove 1-dimensions
x = torch.arange(12).view(3, 4)
print(x.shape)

x = x.unsqueeze(dim=1)
print(x.shape)

x = x.squeeze()
print(x.shape)

# all of the standard mathematics operations apply
x = torch.rand(3, 4)
print("x: \n", x)
print('-' * 20)
print("torch.add(x, x): \n", torch.add(x, x))
print("-" * 20)
print("x + x: \n", x + x)

# The convention of _ indicating in-place operations continues
x = torch.arange(12).reshape(3, 4)
print(x)
print(x.add_(x))

# There are many operations for which reduce a dimension. Such as sum
x = torch.arange(12).reshape(3, 4)
print("x: \n", x)
print("--------------------")
print("Summing across rows (dim=0): \n", x.sum(dim=0))
print("--------------------")
print("Summing across columns (dim=1): \n", x.sum(dim=1))

# Indexing, Slicing, Joining and Mutating
x = torch.arange(6).view(2, 3)
print("x: \n", x)
print("-" * 20)
print("x[:2][:2]: \n", x[:2][:2])
print("-" * 20)
print("x[0][1]: \n", x[0][1])
print("-" * 20)
print("Setting [0][1] to be 8")
x[0][1] = 8
print(x)

# We can select a subset of a tensor using the index_select
x = torch.arange(9).view(3, 3)
print(x)

print("-" * 20)
indices = torch.LongTensor([0, 2])  # row select
print(torch.index_select(x, dim=0, index=indices))

print("-" * 20)
indices = torch.LongTensor([0, 2])  # columns select
print(torch.index_select(x, dim=1, index=indices))

# also use numpy-style advanced indexing
x = torch.arange(9).view(3, 3)
indices = torch.LongTensor([0, 2])

print(x[indices])
print("-" * 20)
print(x[indices, :])
print("-" * 20)
print(x[:, indices])

# We can combine tensors by concatenating them. First, concatenating on the rows
x = torch.arange(6).view(2, 3)
describe(x)
describe(torch.cat([x, x], dim=0))
describe(torch.cat([x, x], dim=1))
describe(torch.stack([x, x]))

# We can concentate along the first dimension(the columns direction)
x = torch.arange(9).view(3, 3)
print(x)
print("-" * 20)
new_x = torch.cat([x, x, x], dim=1)
print(new_x.shape)
print(new_x)

# We can also concatenate on a new 0th dimension to "stack" the tensors
x = torch.arange(9).view(3, 3)
print(x)
print("-" * 20)
new_x = torch.stack([x, x, x])
print(new_x.shape)
print(new_x)

# Transposing allows you to switch the dimensions to be on different axis
x = torch.arange(0, 12).view(3, 4)
print("x: \n", x)
print("-" * 20)
print("x.tranpose(1, 0): \n", x.transpose(1, 0))

# A 3D tensor would represent a batch of sequences, where each sequence item has a feature vector.It is common to switch the batch and sequence dimensions so that we can more easily index the sequence in a sequence model.
batch_size = 3
seq_size = 4
feature_size = 5
x = torch.arange(batch_size * seq_size *
                 feature_size).view(batch_size, seq_size, feature_size)
print("x.shape: \n", x.shape)
print("x: \n", x)
print("-" * 20)
print("x.transpose(1,0).shape: \n", x.transpose(
    1, 0).shape)  # 交换axis=0和axis=1两个轴
print("x.transpose(1,0): \n", x.transpose(1, 0))

# permute is a more general version of transpose
batch_size = 3
seq_size = 4
feature_size = 5
x = torch.arange(batch_size * seq_size *
                 feature_size).view(batch_size, seq_size, feature_size)
print("x.shape: \n", x.shape)
print("x: \n", x)
print("-" * 20)
print("x.permute(1, 0, 2).shape: \n", x.permute(1, 0, 2).shape)  # 交换第一个和第二个轴
print("x.permute(1, 0, 2): \n", x.permute(1, 0, 2))

# matrix multipy is mm
x = torch.randn(2, 3, requires_grad=True)
print(x)

x1 = torch.arange(6).view(2, 3).float()
describe(x1)
x2 = torch.ones(3, 2)
print("original x2: \n", x2)
x2[:, 1] += 1
describe(x2)
describe(torch.mm(x1, x2))  # matrix multipy
print("-" * 20)
# other ways
x = torch.arange(0, 12).view(3, 4).float()
print("x: \n", x)
x2 = torch.ones(4, 2)
x2[:, 1] += 1
print("x2: \n", x2)
print(x.mm(x2))

# compute gradient
x = torch.tensor([[2.0, 3.0]], requires_grad=True)
z = 3 * x  # a tensor and multipy it by 3
print("z: \n", z)

loss = z.sum()  # create a scalar output using sum(),A scalar output is needed as the loss variable
print("loss: \n", loss)

loss.backward()  # compute loss respect to inputs rate of change
print("after loss.backward(),x.grad: \n", x.grad)

# example:compute a conditional function gradient


def f(x):
    if(x.data > 0).all():
        return torch.sin(x)
    else:
        return torch.cos(x)


x = torch.tensor([1.0], requires_grad=True)
y = f(x)
y.backward()
print("x.grad: \n", x.grad)
# We could apply this to a larger vector too, but we need to make sure the output is a scalar
x = torch.tensor([1.0, 0.5], requires_grad=True)
y = f(x)
y.sum().backward()  # when tensor is a vector,y must be a scalar!
print("x.grad: \n", x.grad)

# this is not right for this edge case, because we aren't doing the boolean computation and subsequent application of cos and sin on an elementwise basis
x = torch.tensor([1.0, -1], requires_grad=True)
y = f(x)
y.sum().backward()
print("x.grad: \n", x.grad)

# slove this problem


def f2(x):
    mask = torch.gt(x, 0).float()
    return mask * torch.sin(x) + (1 - mask) * torch.cos(x)


x = torch.tensor([1.0, -1], requires_grad=True)
y = f2(x)
y.sum().backward()
print("x.grad: \n", x.grad)

# describe grad


def describe_grad(x):
    if x.grad is None:
        print("No gradient information")
    else:
        print("Gradient:{}".format(x.grad))
        print("Gradient Function:{}".format(x.grad_fn))


x = torch.ones(2, 2, requires_grad=True)
describe(x)
describe_grad(x)
print("-" * 20)
y = (x + 2) * (x + 5) + 3
describe(y)
z = y.mean()
describe(z)
describe_grad(x)
print("-" * 20)
z.backward(create_graph=True, retain_graph=True)
describe_grad(x)
print("-" * 20)

x = torch.ones(2, 2, requires_grad=True)
y = x + 2
print(y.grad_fn)
y.sum().backward()
print(x.grad)

# CUDA tensor(use on GPU)
print(torch.cuda.is_available())
x = torch.rand(3, 3)
describe(x)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

x = torch.rand(3, 3).to(device)
describe(x)
print("device:", x.device)

cpu_device = torch.device("cpu")
y = torch.rand(3, 3)
z = x + y
print(z)

# convert torch.cuda.tensor to torch.tensor(if you code running on GPU)
x = x.to(cpu_device)
y = y.to(cpu_device)
z = x + y
print(z)
"""
if torch.cuda.is_available():  # only is GPU is available
    a = torch.rand(3, 3).to(device="cuda:0")  #  CUDA Tensor
    print(a)
    b = torch.rand(3, 3).cuda()
    print(b)
    print(a + b)
    a = a.cpu()
    print(a + b)
"""

# Exercises

# Create a 2D tensor and then add a dimension of size 1 inserted at the 0th axis.
x = torch.rand(2, 2)
describe(x)
x = x.unsqueeze(dim=0)
describe(x)
# Remove the extra dimension you just added to the previous tensor.
x = x.squeeze(dim=0)
describe(x)
# Create a random tensor of shape 5x3 and all elements range set in  [3, 7)
x = 3 + torch.rand(5, 3) * 4
describe(x)
# Create a tensor with values from a normal distribution (mean=0, std=1).
a = torch.rand(3, 3)
a = a.normal_(mean=0, std=1)
print(a)
# Retrieve the indexes of all the non zero elements in the tensor torch.Tensor([1, 1, 1, 0, 1]).
x = torch.tensor([1, 1, 1, 0, 1])
nonzero_index = torch.nonzero(x)  # output nonzero element index
print(nonzero_index)
# Create a random tensor of size (3,1) and then horizonally stack 4 copies together.
x = torch.rand(3, 1)
x = x.expand(3, 4)
print(x)
# Return the batch matrix-matrix product of two 3 dimensional matrices (a=torch.rand(3,4,5), b=torch.rand(3,5,4)).
a = torch.rand(3, 4, 5)
b = torch.rand(3, 5, 4)
c = torch.bmm(a, b)
print(c)
