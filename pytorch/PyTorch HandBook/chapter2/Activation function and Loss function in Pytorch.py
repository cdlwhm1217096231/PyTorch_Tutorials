#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-02-23 14:35:28
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(1337)
torch.cuda.manual_seed_all(1337)
np.random.seed(1337)


# 1.implementing a perceptron using PyTorch


class Percetron(nn.Module):
    """
    A Percetron is one linear layer
    """

    def __init__(self, input_dim):
        """
        input_dim:the dimension of the input features
        """
        super(Percetron, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)

    def forward(self, x_in):
        """
        x_in:an input tensor
        x_in.shape:(batch, num_features)
        return:
            (batch,)
        """
        return nn.sigmoid(self.fc1(x_in)).squeeze()


percetron = Percetron(3)
print(percetron)


# 2.implementing a sigmoid activation function

x = torch.arange(-5, 5, 0.1)
y = torch.sigmoid(x)
plt.plot(x.numpy(), y.detach().numpy())
plt.title("sigmoid function")
plt.show()

# detach() introduction
"""
返回一个新的从当前图中分离的 Variable
返回的 Variable 永远不会需要梯度
如果 被 detach 的Variable volatile=True， 那么 detach 出来的 volatile 也为 True
返回的 Variable 和 被 detach 的Variable 指向同一个 tensor
"""
t1 = torch.FloatTensor([1., 2.])
v1 = Variable(t1)
t2 = torch.FloatTensor([3., 4.])
v2 = Variable(t2)
v3 = v1 + v2
v3_detached = v3.detach()
v3_detached.data.add_(t1)  # 修改了v3_detached Variable中的tensor值
print("v3=", v3)
print("v3_detached=", v3_detached)

# detach_() introduction
"""
作用：将 Variable 从创建它的 graph 中分离，把它作为叶子节点
用途：如果我们有两个网络 A,B.A,B, 两个关系是这样的 y=A(x),z=B(y)y=A(x),z=B(y) 现在我们想用 z.backward()z.backward() 来为 BB 网络的参数来求梯度，但是又不想求 AA 网络参数的梯度。
"""

"""
# 方法1
y = A(x)
z = B(y.detach())
z.backward()

# 方法2
y = A(x)
y.detach_()
z = B(y)
z.backward()
"""

# 3.Tanh activation function
x = torch.arange(-5, 5, 0.1)
y = torch.tanh(x)
plt.plot(x.numpy(), y.detach().numpy())
plt.title("Tanh function")
plt.show()

# 4.ReLU activation function
relu = torch.nn.ReLU()
x = torch.arange(-5., 5., 0.1)
y = relu(x)
plt.plot(x.numpy(), y.detach().numpy())
plt.title("ReLU function")
plt.show()

# 5.Softmax function
x_input = torch.rand(1, 3)
softmax = nn.Softmax(dim=1)  # columns direction
y_output = softmax(x_input)
print("x_input:", x_input)
print("y_output:", y_output)
print(torch.sum(y_output, dim=1))

# 6.MSE loss
mse_loss = nn.MSELoss()
outputs = torch.randn(3, 5, requires_grad=True)
targets = torch.randn(3, 5)
loss = mse_loss(outputs, targets)
loss.backward()
print("mse loss = ", loss)

# 7.Cross-Entropy loss
ce_loss = nn.CrossEntropyLoss()
outputs = torch.randn(3, 5, requires_grad=True)
targets = torch.tensor([1, 0, 3], dtype=torch.int64)
loss = ce_loss(outputs, targets)
loss.backward()
print("ce loss = ", loss)

# 8.Binary cross_entropy loss
bce_loss = nn.BCELoss()
sigmoid = nn.Sigmoid()
outputs = sigmoid(torch.randn(4, 1, requires_grad=True))
targets = torch.tensor([1, 0, 1, 0], dtype=torch.float32).view(4, 1)
loss = bce_loss(outputs, targets)
loss.backward()
print("bce loss = ", loss)
