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


'''config'''
class Config:
    def __init__(self, n_folds=5, n_epochs=20, batch_size=16, patience=3,
                 lr=2e-5, max_len_char=410, ways_of_mask=2):
        self.n_folds = n_folds
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.lr = lr  # 学习率参考https://github.com/ymcui/Chinese-BERT-wwm
        self.max_len_char = max_len_char  # 411+2  todo
        self.ways_of_mask = ways_of_mask  # dynamic masking
        self.tokenizer = transformers.BertTokenizer.from_pretrained('inputs/chinese-roberta-wwm-ext',
                                                                    do_lower_case=False)
        self.tokenizer.add_tokens(char_lis)  # todo


my_config = Config()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.text = df['content']
        self.char = df["character"]
        self.emo_a = df["emo_a"]
        self.emo_b = df["emo_b"]
        self.emo_c = df["emo_c"]
        self.emo_d = df["emo_d"]
        self.emo_e = df["emo_e"]
        self.emo_f = df["emo_f"]
        self.tokenizer = my_config.tokenizer

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = self.text[item]
        char = self.char[item]
        emo_a = self.emo_a[item]
        emo_b = self.emo_b[item]
        emo_c = self.emo_c[item]
        emo_d = self.emo_d[item]
        emo_e = self.emo_e[item]
        emo_f = self.emo_f[item]

        encoded_inputs = self.tokenizer(text,
                                        padding='max_length',
                                        truncation=True,
                                        max_length=my_config.max_len_char)
        if isinstance(char, str):
            char_id = self.tokenizer.convert_tokens_to_ids(char)
            out_pos = encoded_inputs["input_ids"].index(char_id)
        else:
            out_pos = 0


        return {
            "input_ids": torch.tensor(encoded_inputs["input_ids"], dtype=torch.int64),
            "token_type_ids": torch.tensor(encoded_inputs["token_type_ids"], dtype=torch.int64),
            "attention_mask": torch.tensor(encoded_inputs["attention_mask"], dtype=torch.int64),
            "out_pos": torch.tensor(out_pos, dtype=torch.int64),
            "emo_a": torch.tensor(emo_a, dtype=torch.int64),
            "emo_b": torch.tensor(emo_b, dtype=torch.int64),
            "emo_c": torch.tensor(emo_c, dtype=torch.int64),
            "emo_d": torch.tensor(emo_d, dtype=torch.int64),
            "emo_e": torch.tensor(emo_e, dtype=torch.int64),
            "emo_f": torch.tensor(emo_f, dtype=torch.int64)
        }


class Model()