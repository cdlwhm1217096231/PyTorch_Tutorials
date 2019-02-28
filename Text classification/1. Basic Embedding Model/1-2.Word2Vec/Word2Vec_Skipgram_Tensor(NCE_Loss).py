#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-02-28 09:50:10
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
Skipgram + negative sampling,此时的损失函数选用交叉熵损失函数NCE Loss
"""
tf.reset_default_graph()


# 3 Words Sentence
sentences = ["i like dog", "i like cat", "i like animal",
             "dog cat animal", "apple cat dog like", "dog fish milk like",
             "dog cat eyes like", "i like apple", "apple i hate",
             "apple i movie book music like", "cat dog hate", "cat dog like"]

word_sequence = " ".join(sentences).split()
word_list = list(set(word_sequence))
word_dict = {w: i for i, w in enumerate(word_list)}

# model parameters
batch_size = 20
embedding_size = 2  # To show 2 dim embedding graph
num_sampled = 10  # 对于负样本,进行负采样操作,抽取的负样本数量是小于batch_size
vocab_size = len(word_list)


def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        random_inputs.append(data[i][0])  # target
        random_labels.append([data[i][1]])  # context word

    return random_inputs, random_labels


# windows size is 1
skip_grams = []
for i in range(1, len(word_sequence) - 1):
    target = word_dict[word_sequence[i]]
    context = [word_dict[word_sequence[i - 1]],
               word_dict[word_sequence[i + 1]]]

    for w in context:
        skip_grams.append([target, w])

# 建立模型
inputs = tf.placeholder(tf.int32, shape=[batch_size])
# To use tf.nn.nce_loss, [batch_size, 1]
labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

embeddings = tf.Variable(tf.random_uniform(
    [vocab_size, embedding_size], -1.0, 1.0))
selected_embed = tf.nn.embedding_lookup(embeddings, inputs)
# 参考:https://www.jianshu.com/p/abea0d9d2436

nce_weights = tf.Variable(tf.random_uniform(
    [vocab_size, embedding_size], -1.0, 1.0))
nce_biases = tf.Variable(tf.zeros([vocab_size]))

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.nce_loss(
    nce_weights, nce_biases, labels, selected_embed, num_sampled, vocab_size))
"""
假设输入数据是K维，一共有N个类
weight:(N,K)
biases:N
inputs:(batch_size)
labels:(batch_size, num_true)
num_true:实际的正样本个数
num_sampled:采样出的负样本个数
num_classes:N
"""
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# 开始训练
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(5000):
        batch_inputs, batch_labels = random_batch(skip_grams, batch_size)
        _, loss = sess.run([optimizer, cost], feed_dict={
                           inputs: batch_inputs, labels: batch_labels})

        if (epoch + 1) % 1000 == 0:
            print('Epoch:{}'.format(epoch + 1),
                  'cost:{:.6f}'.format(loss))

    trained_embeddings = embeddings.eval()
print("训练完成!")
for i, label in enumerate(word_list):
    x, y = trained_embeddings[i]
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2),
                 textcoords='offset points', ha='right', va='bottom')
plt.show()
