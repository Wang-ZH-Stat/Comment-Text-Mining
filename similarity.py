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

file_list = os.listdir('./data')

def sim_sen(attraction):

    data = pd.read_excel('./data/' + attraction + '.xlsx')

    # abstract = data['景点介绍'].values[0]
    comment = data['评论正文'].values

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

        jieba.add_word(attraction[2:])
        text = peg.cut(text)
        noun = 0
        verb = 0
        adjective = 0
        adverb = 0
        filtered = []
        for word, flag in text:
            if word not in stop_words and len(word) >= 1:
                filtered.append(word)
                if flag == 'n':
                    noun = noun + 1
                if flag == 'v':
                    verb = verb + 1
                if flag == 'a':
                    adjective = adjective + 1
                if flag == 'd':
                    adverb = adverb + 1

        len_fil = len(filtered)
        if len_fil == 0:
            len_fil = 1

        # 停用词
        # filtered = [w for w in text if not w in stop_words and len(w) >= 1]


        return ' '.join(filtered), noun / len_fil, verb / len_fil, adjective / len_fil, adverb / len_fil

    texts_str = []
    noun_list = []
    verb_list = []
    adjective_list = []
    adverb_list = []
    for text in comment:
        text_str, noun, verb, adjective, adverb = textPretreatment(text)
        texts_str.append(text_str)
        noun_list.append(noun)
        verb_list.append(verb)
        adjective_list.append(adjective)
        adverb_list.append(adverb)

    texts = [text_str.split(' ') for text_str in texts_str]

    jieba.analyse.set_stop_words('./百度停用词表.txt')
    jieba.add_word(attraction[2:])
    keywords = jieba.analyse.textrank(' '.join(texts_str), topK=100, allowPOS=('n', 'v', 'a', 'd'))

    # 基于文本集建立词典，并获得词典特征数
    dictionary = Dictionary(texts)
    num_features = len(dictionary.token2id)

    # 3.1、基于词典，将【分词列表集】转换成【稀疏向量集】，称作【语料库】
    corpus = [dictionary.doc2bow(text) for text in texts]
    # 3.2、同理，用【词典】把【搜索词】也转换为【稀疏向量】
    kw_vector = dictionary.doc2bow(keywords)

    # 4、创建【TF-IDF模型】，传入【语料库】来训练
    tfidf = TfidfModel(corpus)

    # 5、用训练好的【TF-IDF模型】处理【被检索文本】和【搜索词】
    tf_texts = tfidf[corpus]  # 此处将【语料库】用作【被检索文本】
    tf_kw = tfidf[kw_vector]

    # 6、相似度计算
    sparse_matrix = SparseMatrixSimilarity(tf_texts, num_features)
    similarities = sparse_matrix.get_similarities(tf_kw)


    # ---------情感
    sentimentslist = []
    for text in texts_str:
        if(len(text) >= 2):
            s = SnowNLP(text)
            sentimentslist.append(s.sentiments)
        else:
            sentimentslist.append(np.mean(np.array(sentimentslist)))


    # -----评分提取
    score = []
    raw_score = data['评分'].values
    for sc in raw_score:
        if(sc=='5分 超棒'):
            score.append(5)
        elif(sc=='4分 满意'):
            score.append(4)
        elif(sc=='3分 不错'):
            score.append(3)
        elif (sc == '2分 一般'):
            score.append(2)
        elif (sc == '1分 不佳'):
            score.append(1)
        else:
            score.append(None)

    time_raw = data['评论时间'].values
    time = []
    for ti in time_raw:
        time.append(ti[:4] + ti[5:7] + ti[8:10])

    name = data['用户名'].values

    level = data['级别'].values

    emotion_df, emotionlist = get_emotion(attraction)

    mydata = {'name': name, 'level': level, 'time': time, 'score': score,
              'noun': noun_list, 'verb': verb_list, 'adjective': adjective_list, 'adverb': adverb_list,
              'similarity': similarities, 'sentiment': sentimentslist, 'emotion': emotionlist}
    mydata = pd.DataFrame(mydata)

    mydata = pd.concat([mydata, emotion_df], axis=1)

    mydata.to_csv('./data2/' + attraction + '.csv', index=False, encoding='utf-8')
    print(attraction + ' done!')

    return mydata, data['总评分'].values[0], data['级别'].values[0], int(data['评论数'].values[0].strip('条点评'))


attraction = []
score = []
level = []
comment = []
length = []
noun = []
verb = []
adjective = []
adverb = []
similarity = []
sentiment = []
emotion = []
positive = []
negative = []
anger = []
disgust = []
fear = []
sadness = []
surprise = []
good = []
happy = []
for file in file_list:
    mydata, temp_score, temp_level, temp_comment = sim_sen(file.strip('.xlsx'))
    attraction.append(file.strip('.xlsx')[2:])
    score.append(temp_score)
    level.append(temp_level)
    comment.append(temp_comment)
    similarity.append(np.mean(mydata['similarity'].values))
    sentiment.append(np.mean(mydata['sentiment'].values))
    emotion.append(np.mean(mydata['emotion'].values))
    length.append(np.mean(mydata['length'].values))

    noun.append(np.mean(mydata['noun'].values))
    verb.append(np.mean(mydata['verb'].values))
    adjective.append(np.mean(mydata['adjective'].values))
    adverb.append(np.mean(mydata['adverb'].values))

    positive.append(np.mean(mydata['positive'].values))
    negative.append(np.mean(mydata['negative'].values))
    anger.append(np.mean(mydata['anger'].values))
    disgust.append(np.mean(mydata['disgust'].values))
    fear.append(np.mean(mydata['fear'].values))
    sadness.append(np.mean(mydata['sadness'].values))
    surprise.append(np.mean(mydata['surprise'].values))
    good.append(np.mean(mydata['good'].values))
    happy.append(np.mean(mydata['happy'].values))

rank = np.array(range(26)[1:])

attraction_data = {'attraction': attraction, 'rank': rank, 'score': score, 'level': level, 'comment': comment,
                   'length': length, 'similarity': similarity,
                   'noun': noun, 'verb': verb, 'adjective': adjective, 'adverb': adverb,
                   'sentiment': sentiment, 'emotion': emotion,
                   'positive': positive, 'negative': negative, 'anger': anger,
                   'disgust': disgust, 'fear': fear, 'sadness': sadness, 'surprise': surprise,
                   'good': good, 'happy': happy}

attraction_data = pd.DataFrame(attraction_data)
attraction_data.to_csv('./attraction_data.csv', index=False, encoding='utf-8')
attraction_data.to_excel('./attraction_data.xlsx', index=False, encoding='utf-8')


