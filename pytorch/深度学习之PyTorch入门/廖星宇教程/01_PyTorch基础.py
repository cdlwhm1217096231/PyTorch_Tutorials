#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Version: python 3.5.2
# Tools: Pycharm 2017.2.2
import torch
import numpy as np
from torch.autograd import Variable
"""
torch是能够运行在GPU上的张量，而numpy中的ndarray只能在cpu上运行
torch中的tensor与numpy中的ndarray可以相互转换
tensor即张量，0维的tenor代表一个点，1维的tensor代表一个向量，二维的tensor代表一个矩阵，多维的tensor代表一个多维的数组
"""
# 定义矩阵
a = torch.Tensor([[2, 3], [4, 8], [7, 9]])
print("a is\n-->{}".format(a))
print("a size is-->{}".format(a.size()))
# torch.Tensor默认的是torch.FloatTensor类型,也可以自己指定数据类型
b = torch.LongTensor([[2, 3], [4, 8], [7, 9]])
print("b=\n", b)
# 也可以指定全0的空tensor或者是取一个正态分布作为随机初始值
c = torch.zeros((3, 2))
print("c=\n", c)

d = torch.randn(3, 2)
print("d=\n", d)
# 通过像numpy索引的方法取得tensor中的值，并可以进行修改
a[1][0] = 100
print("a=\n", a)
"""重点！tensor与ndarray之间的转换"""
# tensor转为numpy
numpy_b = b.numpy()
print("numpy表示的b:\n", numpy_b)
# numpy转为tensor
e = np.array([[2, 3], [4, 5]])
tensor_e = torch.from_numpy(e)
print("tensor表示的e:\n", tensor_e)
# 改变tensor_e的数据类型
tensor_e_float = tensor_e.float()
print(tensor_e_float)
# 判断自己的电脑是否支持GPU加速，如果想把tensor a 放到GPU上，只需要a.cuda()就能将tensor a放到GPU上
if torch.cuda.is_available():
    a_cuda = a.cuda()
    print(a_cuda)
"""变量Variable"""
# 神经网络计算图中特有的，Variable提供了自动求导的功能，
# Variable与Tensor本质上没有区别，不过Variable会放入一个计算图中，然后进行前向传播、反向传播、自动求导

# 将一个tensor转化为Variable
a_v = Variable(a)
print("a_v=\n", a_v)
# Variable有3个重要的属性：data  grad    grad_fn
"""
data:通过data取出存放在Variable中的tensor数值
grad: Variable的反向传播速度
grad_fn: 得到这个Variable的操作
"""
# 创建变量
x = Variable(torch.Tensor([1]), requires_grad=True)  # requires_grad参数表示是否对这个变量求梯度，默认是False
w = Variable(torch.Tensor([2]), requires_grad=True)
b = Variable(torch.Tensor([3]), requires_grad=True)
# 构建计算图
y = w * x + b
# 计算梯度
y.backward()  # 自动求导
# 打印出梯度
print(x.grad, w.grad, b.grad)
# 上面的例子是针对一个标量求导，下面是对矩阵求导
x = torch.randn(3)
x = Variable(x, requires_grad=True)
y = x * 2
print(y)
y.backward(torch.FloatTensor([1, 1, 1]))   # 得到的y是一个三维的向量，需要传入参数声明
print(x.grad)
"""Dataset数据集"""
# torch.utils.data.DataLoader来定义一个新的迭代器，进行数据的读取与预处理
# torch.utils.data.DataLoader(myDataSet, batch_size=32, shuffle=True, collate_fn=default_collate)   collate_fn表示如何取样本
# torchvision包中还定义了ImageFolder，主要是处理图片，dset = ImageFolder(root="root_path", transform=None, loader=default_loader)

"""nn.Module模组"""
# pytorch编写神经网络时，所有的层结构和损失函数都来自于torch.nn，所有的模型构建都继承基类nn.Model
"""
class net_name(nn.Model):
    def __init__(self, other_arguments):
        super(net_name, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)
        # 其他层
    
    def forward(self, x):
        x = self.conv1(x)
        return x
"""
# 定义完模型后，通过nn这个包来定义损失函数
"""
loss_func = nn.CrossEntropyLoss()
loss = loss_fun(y_hat, y)
"""
# 优化torch.optim
"""optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)"""
# 注意！在进行优化前需要先将梯度清0，即optimizer.zeros(),然后通过loss.backward()反向传播，自动求导得到每个参数的梯度
# 最后，optimizer.step()就可以通过梯度来进行参数的更新

"""模型的保存与加载"""
# 1.保存整个模型的结构信息与参数信息，保存的对象是模型model  torch.save(model, "./model.pth")
# 2.保存模型的参数，保存的对象是模型的状态，model.state_dict()  torch.save(model.state_dict(), "./model_state.pth")
# 1.加载完整的模型结构与参数信息  load_model = torch.load("model.pth")
# 2.加载模型参数信息，需要先导入模型的结构,然后通过model.load_state_dict(torch.load("model_state.pth"))来导入

