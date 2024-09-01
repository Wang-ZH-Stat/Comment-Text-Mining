# coding: utf-8
import pandas as pd
import jieba
import time
import csv
from dict import *

# attraction = '04盘山'

def get_emotion(attraction):
    data = pd.read_excel('./data/' + attraction + '.xlsx')

    # 扩展前的词典
    df = pd.read_excel('大连理工大学中文情感词汇本体NAU.xlsx')

    df = df[['词语', '词性种类', '词义数', '词义序号', '情感分类', '强度', '极性']]
    df.head()

    Happy = []
    Good = []
    Surprise = []
    Anger = []
    Sad = []
    Fear = []
    Disgust = []

    # df.iterrows()功能是迭代遍历每一行
    for idx, row in df.iterrows():
        if row['情感分类'] in ['PA', 'PE']:
            Happy.append(row['词语'])
        if row['情感分类'] in ['PD', 'PH', 'PG', 'PB', 'PK']:
            Good.append(row['词语'])
        if row['情感分类'] in ['PC']:
            Surprise.append(row['词语'])
        if row['情感分类'] in ['NB', 'NJ', 'NH', 'PF']:
            Sad.append(row['词语'])
        if row['情感分类'] in ['NI', 'NC', 'NG']:
            Fear.append(row['词语'])
        if row['情感分类'] in ['NE', 'ND', 'NN', 'NK', 'NL']:
            Disgust.append(row['词语'])
        if row['情感分类'] in ['NAU']:  # 修改: 原NA算出来没结果
            Anger.append(row['词语'])

        # 正负计算不是很准 自己可以制定规则
    Positive = Happy + Good + Surprise
    Negative = Anger + Sad + Fear + Disgust

    # ---------------------------------------中文分词---------------------------------

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
    def textPretreatment(text, List=True):
        text = str(text)
        text = reserve_chinese(text)
        # 分词（加入领域词典）
        jieba.add_word('盘山')
        text = jieba.cut(text, cut_all=False)
        # 停用词
        filtered = [w for w in text if not w in stop_words and len(w) >= 1]

        if (List == True):
            return filtered
        else:
            return ' '.join(filtered)

    # 情感统计
    def emotion_caculate(text):
        positive = 0
        negative = 0

        anger = 0
        disgust = 0
        fear = 0
        sad = 0
        surprise = 0
        good = 0
        happy = 0

        anger_list = []
        disgust_list = []
        fear_list = []
        sad_list = []
        surprise_list = []
        good_list = []
        happy_list = []

        wordlist = textPretreatment(text)
        # wordlist = jieba.lcut(text)
        wordset = set(wordlist)
        wordfreq = []
        for word in wordset:
            freq = wordlist.count(word)
            tlist = []
            if word in Positive:
                positive += freq
            if word in Negative:
                negative += freq
            if word in Anger:
                anger += freq
                anger_list.append(word)
                tlist.append("anger")
                tlist.append(word)
                tlist.append(freq)
            if word in Disgust:
                disgust += freq
                disgust_list.append(word)
                tlist.append("disgust")
                tlist.append(word)
                tlist.append(freq)
            if word in Fear:
                fear += freq
                fear_list.append(word)
                tlist.append("fear")
                tlist.append(word)
                tlist.append(freq)
            if word in Sad:
                sad += freq
                sad_list.append(word)
                tlist.append("sad")
                tlist.append(word)
                tlist.append(freq)
            if word in Surprise:
                surprise += freq
                surprise_list.append(word)
                tlist.append("surprise")
                tlist.append(word)
                tlist.append(freq)
            if word in Good:
                good += freq
                good_list.append(word)
                tlist.append("good")
                tlist.append(word)
                tlist.append(freq)
            if word in Happy:
                happy += freq
                happy_list.append(word)
                tlist.append("happy")
                tlist.append(word)
                tlist.append(freq)

        len_emo = positive + negative
        if len_emo == 0:
            len_emo = 1
        emotion_info = {
            'length': len(wordlist),
            'positive': positive / len_emo,
            'negative': negative / len_emo,
            'anger': anger / len_emo,
            'disgust': disgust / len_emo,
            'fear': fear / len_emo,
            'good': good / len_emo,
            'sadness': sad / len_emo,
            'surprise': surprise / len_emo,
            'happy': happy / len_emo,
        }

        indexs = ['length', 'positive', 'negative', 'anger', 'disgust', 'fear', 'sadness', 'surprise', 'good', 'happy']
        # return pd.Series(emotion_info, index=indexs), anger_list, disgust_list, fear_list, sad_list, surprise_list, good_list, happy_list
        return pd.Series(emotion_info, index=indexs)


    # ---------------------------------------情感计算---------------------------------
    emotion_df = data['评论正文'].apply(emotion_caculate)


    sentiment_dict_path = "sentiment_words_chinese.tsv"
    degree_dict_path = "degree_dict.txt"
    stop_dict_path = "百度停用词表.txt"

    score = Score(sentiment_dict_path, degree_dict_path, stop_dict_path)

    emotion = []
    for temp in data['评论正文'].values:
        words = textPretreatment(temp)
        words_ = score.remove_stopword(words)

        # 分词->情感词间是否有否定词/程度词+前后顺序->分数累加
        result = score.get2score_position(words_)
        emotion.append(result)

    return emotion_df, emotion
