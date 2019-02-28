#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-02-28 09:00:29
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


tf.reset_default_graph()

# 3 Words Sentence
sentences = ["i like dog", "i like cat", "i like animal",
             "dog cat animal", "apple cat dog like", "dog fish milk like",
             "dog cat eyes like", "i like apple", "apple i hate",
             "apple i movie book music like", "cat dog hate", "cat dog like"]

word_sequence = " ".join(sentences).split()
word_list = list(set(word_sequence))
word_dict = {w: i for i, w in enumerate(word_list)}
print(word_dict)

# model parameters
batch_size = 20
embedding_size = 2
vocab_size = len(word_list)


def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        random_inputs.append(np.eye(vocab_size)[data[i][0]])
        random_labels.append(np.eye(vocab_size)[data[i][1]])
    return random_inputs, random_labels


# windows size=1
skip_grams = []
for i in range(1, len(word_sequence) - 1):
    center_word = word_dict[word_sequence[i]]
    context_words = [word_dict[word_sequence[i - 1]],
                     word_dict[word_sequence[i + 1]]]
    for word in context_words:
        skip_grams.append([center_word, word])


# model
inputs = tf.placeholder(tf.float32, [None, vocab_size])
labels = tf.placeholder(tf.float32, [None, vocab_size])

# W
W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
# WO
WO = tf.Variable(tf.random_uniform([embedding_size, vocab_size], -1.0, 1.0))

hidden_layer = tf.matmul(inputs, W)   # [batch_size, embedding_size]
output_layer = tf.matmul(hidden_layer, WO)  # [batch_size, vocab_size]

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=output_layer, labels=labels))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(5000):
        batch_inputs, batch_targets = random_batch(skip_grams, batch_size)
        _, loss = sess.run([optimizer, cost], feed_dict={
                           inputs: batch_inputs, labels: batch_targets})
        if (epoch + 1) % 1000 == 0:
            print("Epoch:{}".format(epoch + 1), "Loss:{:.4f}".format(loss))

        trained_embeddings = W.eval()   # 画图程序
    print("训练完成!")
for i, label in enumerate(word_list):
    x, y = trained_embeddings[i]
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2),
                 textcoords='offset points', ha='right', va='bottom')
plt.show()
