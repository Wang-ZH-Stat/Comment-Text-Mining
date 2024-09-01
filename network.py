import numpy as np
import pandas as pd
import jieba
import jieba.analyse
from snownlp import SnowNLP
from scipy.sparse import coo_matrix
import csv

data = pd.read_excel('./data/04盘山.xlsx')

raw_text = data['评论正文'].values
raw_text = ' '.join(raw_text)

s = SnowNLP(raw_text)
s.keywords(limit=20)
s.summary(limit=5)

stop_words = []
with open('./百度停用词表.txt', 'r', encoding='utf-8') as f2:
    for stop_word in f2.readlines():
        stop_words.append(stop_word.strip('\n'))

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

comment = data['评论正文'].values
texts = [textPretreatment(text) for text in comment]

jieba.add_word('盘山')
keywords = jieba.analyse.textrank(' '.join(texts), topK=50, allowPOS=('n', 'v', 'a', 'd'))

word_vector = coo_matrix((len(keywords), len(keywords)), dtype=np.int8).toarray()

for text in texts:
    nums = text.split(' ')
    # 循环遍历关键词所在位置 设置word_vector计数
    i = 0
    j = 0
    while i < len(nums):
        j = i + 1
        w1 = nums[i]  # 第一个单词
        while j < len(nums):
            w2 = nums[j]  # 第二个单词
            # 从word数组中找到单词对应的下标
            k = 0
            n1 = 0
            while k < len(keywords):
                if w1 == keywords[k]:
                    n1 = k
                    break
                k = k + 1
            # 寻找第二个关键字位置
            k = 0
            n2 = 0
            while k < len(keywords):
                if w2 == keywords[k]:
                    n2 = k
                    break
                k = k + 1
            # 词频矩阵赋值，只计算上三角
            if n1 <= n2:
                word_vector[n1][n2] = word_vector[n1][n2] + 1
            else:
                word_vector[n2][n1] = word_vector[n2][n1] + 1
            j = j + 1
        i = i + 1

# --------------------------第四步  CSV文件写入--------------------------
with open("./data3/共现矩阵.csv", "w", encoding='utf-8-sig', newline='') as f4:
    writer = csv.writer(f4)
    writer.writerow(['词语1', '词语2', '共现次数'])

    i = 0
    while i < len(keywords):
        w1 = keywords[i]
        j = 0
        while j < len(keywords):
            w2 = keywords[j]
            if word_vector[i][j] > 4:
                templist = []
                templist.append(w1)
                templist.append(w2)
                templist.append(str(int(word_vector[i][j])))
                writer.writerow(templist)
            j = j + 1
        i = i + 1

# 观察共现次数最多的20组词
df = pd.read_csv('./data3/共现矩阵.csv', encoding='utf-8-sig')
sort_df = df.sort_values(axis=0, ascending=False, by='共现次数')
sort_df.head(20)

# ----------------------第五步 准备构建知识图谱需要的两个文件----------------------------
with open("./data3/relationship.csv", "w", encoding='utf-8', newline='') as f5:
    writer = csv.writer(f5)  # 写入对象
    writer.writerow(['Source', 'Target', 'Type', 'Weight'])
    i = 0
    while i < len(keywords):
        w1 = keywords[i]
        j = 0
        while j < len(keywords):
            w2 = keywords[j]
            if word_vector[i][j] > 4:
                # 写入文件
                templist = []
                templist.append(w1)
                templist.append(w2)
                templist.append('Undirected')
                templist.append(str(int(word_vector[i][j])))
                writer.writerow(templist)
            j = j + 1
        i = i + 1

w1_df = df['词语1'].to_list()
w2_df = df['词语2'].to_list()
node_word = []
for w in w1_df:
    if w not in node_word:
        node_word.append(w)
for w in w2_df:
    if w not in node_word:
        node_word.append(w)
entity = {'id': node_word, 'label': node_word}
entity = pd.DataFrame(entity)
entity.to_csv("./data3/entity.csv", encoding='utf-8', index=0)


