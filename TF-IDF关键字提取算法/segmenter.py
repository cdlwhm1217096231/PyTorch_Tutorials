#!/usr/bin/python
# -*- coding: utf-8 -*-

import jieba
import re


"""jieba分词器"""


def segment(sentence, cut_all=False):
    sentence = sentence.replace('\n', '').replace(
        '\u3000', '').replace('\u00A0', '')  # 使用空格替换不同语言中的空格
    sentence = ' '.join(jieba.cut(sentence, cut_all=cut_all))  # 全模式进行分词
    return re.sub('[a-zA-Z0-9.。:：,，)）(（！!??”“\"]', '', sentence).split()


"""
1.不间断空格\u00A0,主要用在office中,让一个单词在结尾处不会换行显示,快捷键ctrl+shift+space ;
2.半角空格(英文符号)\u0020,代码中常用的;
3.全角空格(中文符号)\u3000,中文文章中使用;
"""
