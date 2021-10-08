'''
(1)任务：剧本角色情感识别
(2)评价指标RMSE
(3)train_dataset_v2.tsv: 以英文制表符分隔
    1)有些样本有角色但没有情感【也就是情感为空】
    2)有些样本没有角色但有情感【包括情感全为0】
    3)有些样本没有角色也没有情感【也就是情感为空】
    4)情感取值范围[0, 1, 2, 3]

'''

import sys  # todo
sys.path.append('../input/data-iqiyi')
import numpy as np
import pandas as pd
import random
import time
import os
import json
import pickle
import tqdm
import logging

import jieba
import re
# from zhon.hanzi import punctuation  # 中文标点符号
# import string  # string.punctuation是英文标点符号

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers

from collections import Counter
from pytorchtools import EarlyStopping  # 别人写好的EarlyStopping
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


train_df = pd.read_csv('../input/data-iqiyi/train_dataset_v2.tsv',
                       sep='\t', header=0, index_col=None)  # todo
test_df = pd.read_csv('../input/data-iqiyi/test_dataset.tsv',
                      sep='\t', header=0, index_col=None)

# 训练集和测试集的所有角色
char_set_train = set(train_df["character"].unique().astype(str).tolist())
char_set_test = set(test_df["character"].unique().astype(str).tolist())
char_set = char_set_train.union(char_set_test)
char_set.remove('nan')
char_lis = list(char_set)
print(len(char_lis))  # 角色总数
with open('./outputs/char_lis', 'wb') as f:
    pickle.dump(char_lis, f)


# # '''EDA 探索数据分析'''
# # 文本长度分布(char级别)
# text_len_train = train_df['content'].map(len)
# print(text_len_train.quantile([0, 0.25, 0.5, 0.75, 1]))  # 最短为2，最长为412
# text_len_test = test_df['content'].map(len)
# print(text_len_test.quantile([0, 0.25, 0.5, 0.75, 1]))  # 最短为3，最长为343
#
# # 角色统计
# char_count_train = train_df['character'].value_counts(ascending=False)
# print(char_count_train.shape[0])  # 角色数
# print(char_count_train.head())
# print(char_count_train.tail())
# char_count_test = test_df['character'].value_counts(ascending=False)
# print(char_count_test.shape[0])  # 角色数
# print(char_count_test.head())
# print(char_count_test.tail())


# 删除情感为空的样本
drop_idx_train = [i for i, j in enumerate(train_df['emotions']) if type(j) != type('str')]
train_df.drop(index=drop_idx_train, inplace=True)
train_df.reset_index(drop=True, inplace=True)

# 处理emotions
emo_train = train_df['emotions'].str.split(',', expand=True)
emo_train.columns = ['emo_a', 'emo_b', 'emo_c', 'emo_d', 'emo_e', 'emo_f']
emo_train['emo_a'] = pd.to_numeric(emo_train['emo_a'])
emo_train['emo_b'] = pd.to_numeric(emo_train['emo_b'])
emo_train['emo_c'] = pd.to_numeric(emo_train['emo_c'])
emo_train['emo_d'] = pd.to_numeric(emo_train['emo_d'])
emo_train['emo_e'] = pd.to_numeric(emo_train['emo_e'])
emo_train['emo_f'] = pd.to_numeric(emo_train['emo_f'])
train_df = pd.concat([train_df, emo_train], axis=1)

cont_char_map_train = {}
for cont, char in zip(train_df["content"], train_df["character"]):
    if cont not in cont_char_map_train:
        cont_char_map_train[cont] = []
    if isinstance(char, str):
        cont_char_map_train[cont].append(char)

cont_char_map_test = {}
for cont, char in zip(test_df["content"], test_df["character"]):
    if cont not in cont_char_map_test:
        cont_char_map_test[cont] = []
    if isinstance(char, str):
        cont_char_map_test[cont].append(char)

# # 在角色前后添加空格，便于tokenizer划分
# def add_space_train(text):
#     for char in cont_char_map_train[text]:
#         text = text.replace(char, ' '+char+' ')
#     return text
# def add_space_test(text):
#     for char in cont_char_map_test[text]:
#         text = text.replace(char, ' '+char+' ')
#     return text
#
# train_df['content'] = train_df['content'].map(add_space_train)
# test_df['content'] = test_df['content'].map(add_space_test)

def add_space(text):
    for char in char_lis:
        text = text.replace(char, ' '+char+' ')
    return text
train_df['content'] = train_df['content'].map(add_space)
test_df['content'] = test_df['content'].map(add_space)

train_df.to_csv('./outputs/train_df.csv', header=True, index=False)
test_df.to_csv('./outputs/test_df.csv', header=True, index=False)

