from jieba import lcut
from gensim.similarities import SparseMatrixSimilarity
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import numpy as np
import pandas as pd
import jieba
import jieba.analyse
from snownlp import SnowNLP
import os
from emotion import *
import jieba.posseg as peg
from collections import Counter
from pyecharts import options as opts
from pyecharts.charts import WordCloud
from pyecharts.globals import SymbolType

text_data = pd.read_excel('./data/04盘山.xlsx')
score_data = pd.read_csv('./data2/04盘山.csv')

pos_text = text_data[score_data['emotion'] > 0]['评论正文'].values
neg_text = text_data[score_data['emotion'] < 0]['评论正文'].values

stop_words = []
with open('./百度停用词表.txt', 'r', encoding='utf-8') as f2:
    for stop_word in f2.readlines():
        stop_words.append(stop_word.strip('\n'))

stop_words.extend(['景区', '景点', '盘山', '不错'])

# 只保留中文
def is_chinese(uchar):
    if uchar >= '\u4e00' and uchar <= '\u9fa5':
        return True
    else:
        return False
def reserve_chinese(content):
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str += i
    return content_str

# 文本预处理
def textPretreatment(text):
    text = str(text)
    text = reserve_chinese(text)
    # 分词（加入领域词典）
    jieba.add_word('盘山')
    text = jieba.cut(text, cut_all=False)
    # 停用词
    filtered = [w for w in text if not w in stop_words and len(w) >= 1]

    return ' '.join(filtered)


def ciyun(text_):
    texts = [textPretreatment(text) for text in text_]
    all_words = ' '.join(texts)
    all_words = all_words.split()
    c = Counter()
    for word in all_words:
        if len(word) > 1 and word != '\r\n':
            c[word] += 1

    # 输出词频最高的前20个词
    high_word = []
    high_num = []

    print('高频词统计结果：')
    for (word, num) in c.most_common(20):
        high_word.append(word)
        high_num.append(num)
        print("%s:%d" % (word, num))


    # ------------------------------------疫情词云分析------------------------------------
    words = []
    for (word, num) in c.most_common(1000):
        words.append((word, num))

    # 渲染图
    def wordcloud_base() -> WordCloud:
        cloud = (
            WordCloud()
            .add("", words, word_size_range=[20, 100], shape=SymbolType.ROUND_RECT)
            .set_global_opts(title_opts=opts.TitleOpts(title='负面情感评论词云图'))
        )
        return cloud

    # 生成图
    wordcloud_base().render('./data3/负面词云图.html')


ciyun(pos_text)
ciyun(neg_text)
