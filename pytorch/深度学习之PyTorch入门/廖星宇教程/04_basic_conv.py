#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-10 10:25:26
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch

# 读入一张灰度图的图片
im = Image.open("./datasets/cat.png").convert("L")
image = np.array(im, dtype="float32")
# 可视化图片
plt.imshow(image.astype("uint8"), cmap='gray')
plt.show()
#  将图片矩阵转化为pytorch tensor，并适配卷积输入的要求
image = torch.from_numpy(image.reshape((1, 1, image.shape[0], image.shape[1])))

"""卷积层"""
# 使用nn.Conv2d---------一般使用nn.Conv2d()这种卷积形式
conv1 = nn.Conv2d(1, 1, 3, bias=False)  # 定义卷积
sobel_kernel = np.array(
    [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype="float32")  # 定义轮廓检测算子
sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))  # 适配卷积的输入和输出
conv1.weight.data = torch.from_numpy(sobel_kernel)  # 给卷积的kernel赋值
edge1 = conv1(Variable(image))  # 作用在图片上
edge1 = edge1.data.squeeze().numpy()  # 将输出转化为图片的格式
plt.imshow(edge1, cmap="gray")
plt.show()

# 使用F.conv2d
sobel_kernel = np.array(
    [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype="float32")
sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
weight = Variable(torch.from_numpy(sobel_kernel))
edge2 = F.conv2d(Variable(image), weight)
edge2 = edge2.data.squeeze().numpy()
plt.imshow(edge2, cmap="gray")
plt.show()


"""池化层"""
# 使用 nn.MaxPool2d----一般使用nn.MaxPool2d()
pool1 = nn.MaxPool2d(2, 2)
print("池化之前:图片的尺寸 {}x{}".format(image.shape[2], image.shape[3]))
small_im1 = pool1(Variable(image))
small_im1 = small_im1.data.squeeze().numpy()
print("池化之后:图片的尺寸 {}x{}".format(small_im1.shape[0], small_im1.shape[1]))
plt.imshow(small_im1, cmap="gray")
plt.show()

# F.max_pool2d
print("池化之前:图片的尺寸 {}x{}".format(image.shape[2], image.shape[3]))
small_im2 = F.max_pool2d(Variable(image), 2, 2)
small_im2 = small_im2.data.squeeze().numpy()
print("池化之后:图片的尺寸 {}x{}".format(small_im2.shape[0], small_im2.shape[1]))
plt.imshow(small_im2, cmap="gray")
plt.show()
