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


# 去重
train_df_nodup = train_df.drop_duplicates(subset=['content'], ignore_index=True)
test_df_nodup = test_df.drop_duplicates(subset=['content'], ignore_index=True)
df_nodup = pd.concat([train_df_nodup, test_df_nodup], axis=0, join='inner', ignore_index=True)

# 将文本长度填充到128左右
context_train_test = [df_nodup['content'][0]]
for text in df_nodup['content'][1:]:
    pre_len = len(context_train_test[-1])
    cur_len = len(text)
    if cur_len + pre_len < 128:
        context_train_test[-1] += text
    else:
        context_train_test.append(text)

