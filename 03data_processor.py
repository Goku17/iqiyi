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


train_df = pd.read_csv('./outputs/train_df.csv', header=0, index_col=None)
test_df = pd.read_csv('./outputs/test_df.csv', header=0, index_col=None)
with open('./outputs/char_lis', 'rb') as f:
    char_lis = pickle.load(f)


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

cont_char_map_train = {}  # 一个样本中的所有角色
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



