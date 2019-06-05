# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import jieba
import csv
import re
import string


# Read the data
def read_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as csv_f:
        csv_reader = csv.reader(csv_f)
        for r in csv_reader:
            assert len(r) == 3, "====>>>> invaild data format!!!"
            label = r[0]

            title = r[1]
            if title != '':
                title = ' '.join(jieba.cut(remove_punctuation(r[1]), cut_all=False))

            content = r[2]
            if content != '':
                content = ' '.join(jieba.cut(remove_punctuation(r[2]), cut_all=False))

            data.append([label, title, content])

    print("read file of %d lines" % (len(data)))

    return data


def remove_punctuation(s):
    punc_0 = "！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    punc_1 = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"
    punc = punc_0 + punc_1

    translator = str.maketrans("", "", punc)

    return s.translate(translator)


# read training data , segment and save it
data_dir = '/Users/meizu/Work/dataset/china_news'

# process train data
train_data = read_data(os.path.join(data_dir, 'train.csv'))
with open(os.path.join(data_dir, 'train.seg'), 'w', encoding='utf-8') as fo:
    for r in train_data:
        fo.write('##yf##'.join(r) + "\n")

# process test data
test_data = read_data(os.path.join(data_dir, 'train.csv'))
with open(os.path.join(data_dir, 'test.seg'), 'w', encoding='utf-8') as fo:
    for r in test_data:
        fo.write('##yf##'.join(r) + "\n")
