'''
(1)任务：剧本角色情感识别
(2)评价指标RMSE
(3)train_dataset_v2.tsv: 以英文制表符分隔
    1)有些样本有角色但没有情感【也就是情感为空】
    2)有些样本没有角色但有情感【包括情感全为0】
    3)有些样本没有角色也没有情感【也就是情感为空】
    4)情感取值范围[0, 1, 2, 3]

'''


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


train_df = pd.read_csv('./data/train_dataset_v2.tsv', sep='\t', header=0, index_col=None)
test_df = pd.read_csv('./data/test_dataset.tsv', sep='\t', header=0, index_col=None)

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

# 训练集和测试集的所有角色
char_set_train = set(train_df["character"].unique().astype(str).tolist())
char_set_test = set(test_df["character"].unique().astype(str).tolist())
char_set = char_set_train.union(char_set_test)
char_set.remove('nan')
char_lis = list(char_set)
print(len(char_lis))  # 角色总数
with open('./outputs/char_lis', 'wb') as f:
    pickle.dump(char_lis, f)

# 在角色前后添加空格，便于tokenizer划分
def add_space(text):
    for char in char_lis:
        text = text.replace(char, ' '+char+' ')
    return text
train_df['content'] = train_df['content'].map(add_space)
test_df['content'] = test_df['content'].map(add_space)

train_df.to_csv('./outputs/train_df.csv', header=True, index=False)
test_df.to_csv('./outputs/test_df.csv', header=True, index=False)

# 去重
train_df_nodup = train_df.drop_duplicates(subset=['content'], ignore_index=True)
test_df_nodup = test_df.drop_duplicates(subset=['content'], ignore_index=True)
df_nodup = pd.concat([train_df_nodup, test_df_nodup], axis=0, join='inner', ignore_index=True)
# 结合前文
context_train_test = [df_nodup['content'][0]]
for text in df_nodup['content'][1:]:
    pre_len = len(context_train_test[-1])
    cur_len = len(text)
    if cur_len + pre_len < 128:
        context_train_test[-1] += text
    else:
        context_train_test.append(text)


with open('./outputs/context_train_test', 'wb') as f:
    pickle.dump(context_train_test, f)