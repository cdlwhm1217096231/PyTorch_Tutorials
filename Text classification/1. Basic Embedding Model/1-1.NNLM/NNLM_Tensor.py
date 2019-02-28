#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-02-26 21:25:01
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import numpy as np
import tensorflow as tf


tf.reset_default_graph()


sentences = ["i like coffee", "i love curry", "i hate apple"]
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
print(word_list)

word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict)


# Model parameters
n_step = 2
n_hidden = 5


def make_batch(sentences):
    input_batch = []
    target_batch = []
    for sentence in sentences:
        words = sentence.split()
        input = [word_dict[word] for word in words[:-1]]
        target = word_dict[words[-1]]

        input_batch.append(np.eye(n_class)[input])  # np.eye()是单位对角阵
        target_batch.append(np.eye(n_class)[target])

    return input_batch, target_batch


# Model

# [batch_size, number of steps, number of Vocabulary]
X = tf.placeholder(tf.float32, [None, n_step, n_class])
Y = tf.placeholder(tf.float32, [None, n_class])

# [batch_size, n_step * n_class]
input = tf.reshape(X, shape=[-1, n_step * n_class])
H = tf.Variable(tf.random_normal([n_step * n_class, n_hidden]))
d = tf.Variable(tf.random_normal([n_hidden]))
U = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

tanh = tf.nn.tanh(d + tf.matmul(input, H))  # [batch_size, n_hidden]
output = tf.matmul(tanh, U) + b  # [batch_size, n_class]

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
prediction = tf.argmax(output, 1)

# Training
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    input_batch, target_batch = make_batch(sentences)

    for epoch in range(5000):
        _, loss = sess.run([optimizer, cost], feed_dict={
                           X: input_batch, Y: target_batch})
        if (epoch + 1) % 1000 == 0:
            print("Epoch:{}".format(epoch + 1), "Cost:{:.4f}".format(loss))
    # Predict
    predict = sess.run([prediction], feed_dict={X: input_batch})

    # Test
    input = [sentence.split()[:2] for sentence in sentences]
    print([sentence.split()[:2] for sentence in sentences],
          '---->', [number_dict[n] for n in predict[0]])
