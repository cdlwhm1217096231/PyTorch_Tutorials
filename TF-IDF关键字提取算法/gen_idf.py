#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import math
import re
import datetime
import sys
import getopt
from segmenter import segment


"""gen_idf.py：用于逆文档频率的生成"""


class MyDocuments(object):    # memory efficient data streaming
    def __init__(self, dirname):
        self.dirname = dirname
        if not os.path.isdir(dirname):
            print(dirname, '- not a directory!')
            sys.exit()

    def __iter__(self):
        for dirfile in os.walk(self.dirname):
            for fname in dirfile[2]:
                text = open(os.path.join(dirfile[0], fname),
                            'r', encoding='utf-8', errors='ignore').read()
                yield segment(text)   # time consuming


def main(argv):   # idf generator
    inputdir = ''
    outputfile = ''

    usage = 'usage: python gen_idf.py -i <inputdir> -o <outputfile>'
    if len(argv) < 4:
        print(usage)
        sys.exit()
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["idir=", "ofile="])
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)

    for opt, arg in opts:   # 解析参数
        if opt == '-h':
            print(usage)
            sys.exit()
        elif opt in ("-i", "--idir"):
            inputdir = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg

    documents = MyDocuments(inputdir)   # 对语料库中的每一篇文章进行分词处理

    ignored = {'', ' ', '', '。', '：', '，', '）', '（', '！', '?', '”', '“'}  # 被忽略的字符
    id_freq = {}  # 定义用来统计词频的字典
    i = 0
    for doc in documents:  # 遍历语料库中的每一篇文章
        doc = set(x for x in doc if x not in ignored)  # 除去每一篇文章中出现的被忽略的词语
        for x in doc:  # 遍历分词后的每一篇文章中的词语
            id_freq[x] = id_freq.get(x, 0) + 1  # 统计每个词语出现的频率
        if i % 1000 == 0:
            print('Documents processed: ', i, ', time: ',
                  datetime.datetime.now())   # 每处理1000篇文档，所用时间统计
        i += 1

    with open(outputfile, 'w', encoding='utf-8') as f:  # 将最终的结果写入outputfile中
        for key, value in id_freq.items():
            f.write(key + ' ' + str(math.log(i / value, 2)) + '\n')  # 计算IDF的公式


if __name__ == "__main__":
    main(sys.argv[1:])
