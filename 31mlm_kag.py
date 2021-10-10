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
sys.path.append('../input/dataiqiyi')
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


train_df = pd.read_csv('../input/dataiqiyi/train_dataset_v2.tsv',
                       sep='\t', header=0, index_col=None)  # todo
test_df = pd.read_csv('../input/dataiqiyi/test_dataset.tsv',
                      sep='\t', header=0, index_col=None)

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


# 在角色前后添加空格，便于tokenizer划分
def add_space(text):
    for char in char_lis:
        text = text.replace(char, ' '+char+' ')
    return text
train_df['content'] = train_df['content'].map(add_space)
test_df['content'] = test_df['content'].map(add_space)


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



'''config'''
class Config:
    def __init__(self, n_folds=10, n_epochs=5, batch_size=16, patience=3,
                 lr=2e-5, max_len_char=410, ways_of_mask=2):
        self.n_folds = n_folds
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.lr = lr  # 学习率参考https://github.com/ymcui/Chinese-BERT-wwm
        self.max_len_char = max_len_char  # 411+2  todo
        self.ways_of_mask = ways_of_mask  # dynamic masking
        self.tokenizer = transformers.BertTokenizer.from_pretrained('../input/hflchineserobertawwmext',
                                                                    do_lower_case=False)  # todo
        self.tokenizer.add_tokens(char_lis)  # todo


my_config = Config()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.text = df['content']
        self.tokenizer = my_config.tokenizer

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = self.text[item]
        text = self.tokenizer.tokenize(text)
        masked_text = ''
        labels = []
        for char in text:
            r1 = np.random.rand()
            if r1 < 0.15:
                id = self.tokenizer.convert_tokens_to_ids([char])
                labels.append(id[0])
                r2 = np.random.rand()
                if r2 < 0.8:
                    masked_text += '[MASK]'
                elif r2 < 0.9:
                    masked_text += char
                else:
                    char_rand = random.choice(text)
                    masked_text += char_rand
            else:
                masked_text += char
                labels.append(-100)

        encoded_inputs = self.tokenizer(masked_text,
                                        padding='max_length',
                                        truncation=True,
                                        max_length=my_config.max_len_char)
        # 在labels的首尾添加101和102对应的标签；对labels截断补全
        length = len(encoded_inputs['input_ids'])
        if len(labels) > length - 2:
            labels = [-100] + labels[:length - 2] + [-100]
        else:
            pad_length = length - 1 - len(labels)
            labels = [-100] + labels + [-100] * pad_length

        return {
            'input_ids': torch.tensor(encoded_inputs['input_ids'], dtype=torch.int64),
            'token_type_ids': torch.tensor(encoded_inputs['token_type_ids'], dtype=torch.int64),
            'attention_mask': torch.tensor(encoded_inputs['attention_mask'], dtype=torch.int64),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }


'''training：定义一个epoch上的训练'''
def training(model, train_dataloader, optimizer, scheduler, device):
    model.train()
    train_dataloader_tqdm = tqdm.tqdm(train_dataloader)
    losses = 0
    for data in train_dataloader_tqdm:
        outputs = model(input_ids=data['input_ids'].to(device=device),
                        attention_mask=data['attention_mask'].to(device=device),
                        token_type_ids=data['token_type_ids'].to(device=device),
                        labels=data['labels'].to(device=device))
        loss = outputs.loss
        losses += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        train_dataloader_tqdm.set_postfix({'loss': loss.item()})  # 当前batch上平均每个样本的loss
    return losses/len(train_dataloader)  # 一个epoch上平均每个样本的损失


'''evaluating：定义一个epoch上的evaluating'''
def evaluating(model, val_dataloader, device):
    model.eval()
    with torch.no_grad():
        val_dataloader_tqdm = tqdm.tqdm(val_dataloader)
        losses_val = 0
        for data in val_dataloader_tqdm:
            outputs = model(input_ids=data['input_ids'].to(device=device),
                            attention_mask=data['attention_mask'].to(device=device),
                            token_type_ids=data['token_type_ids'].to(device=device),
                            labels=data['labels'].to(device=device))
            loss = outputs.loss
            losses_val += loss.item()
            val_dataloader_tqdm.set_postfix({'loss': loss.item()})
    return losses_val/len(val_dataloader)


def main(df, fold_num, idx_shuffled):
    '''
    :param df: 数据集
    :param fold_num: 将第 fold_num 折作为验证集
    :param idx_shuffled: ndarray, 已经打乱的数据集下标
    '''
    ########## 划分数据集 ##########
    val_size = df.shape[0] // my_config.n_folds  # 验证集大小
    idx_shuffled = idx_shuffled.tolist()  # 转换成列表，方便拼接下标
    if fold_num == my_config.n_folds - 1:
        val_idx = idx_shuffled[val_size * fold_num:]
        train_idx = idx_shuffled[:val_size * fold_num]
    else:
        val_idx = idx_shuffled[val_size * fold_num:val_size * (fold_num + 1)]
        train_idx = idx_shuffled[:val_size * fold_num] + idx_shuffled[val_size * (fold_num + 1):]

    train_df = df.iloc[train_idx, :]
    val_df = df.iloc[val_idx, :]

    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    ########## DataLoader ##########
    train_dataset = Dataset(train_df)
    val_dataset = Dataset(val_df)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=my_config.batch_size,
                                                   shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=my_config.batch_size,
                                                 shuffle=False)

    ########## model ##########
    model_config = transformers.BertConfig.from_pretrained('../input/hflchineserobertawwmext/config.json')  # todo
    model = transformers.BertForMaskedLM.from_pretrained('../input/hflchineserobertawwmext',
                                                         config=model_config)  # 哈工大预训练模型
    model.resize_token_embeddings(len(my_config.tokenizer))  # todo word_embedding.shape=(21151,768)
    model = model.to(device=device)

    ########## optimizer & scheduler ##########
    # 如果参数名中包含'bias'或'LayerNorm.weight'【实际上也包括了'LayerNorm.bias'】，就不对其进行L2正则化(即weight_decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=my_config.lr)

    # num_training_steps = int(train_df.shape[0] / my_config.batch_size * my_config.n_epochs)
    # num_warmup_steps = int(train_df.shape[0] / my_config.batch_size * 0.6)
    # scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
    #                                                          num_warmup_steps=num_warmup_steps,
    #                                                          num_training_steps=num_training_steps)
    scheduler = None

    early_stopping = EarlyStopping(patience=my_config.patience,
                                   verbose=True,
                                   path='./outputs/mlm_checkpoint%d.pt' % fold_num)
    epoch_record = None
    for epoch in range(1, my_config.n_epochs+1):
        epoch_record = epoch
        seed = epoch % my_config.ways_of_mask
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)

        training(model, train_dataloader, optimizer, scheduler, device)
        loss_val = evaluating(model, val_dataloader, device)
        early_stopping(loss_val, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    if early_stopping.early_stop:
        return early_stopping.val_loss_min, epoch_record - my_config.patience
    else:
        return early_stopping.val_loss_min, my_config.n_epochs


if __name__ == '__main__':
    df = pd.DataFrame(context_train_test, columns=['content'])  # 使用所有数据集
    idx_shuffled = np.random.permutation(df.shape[0])  # 保证只打乱一次下标

    val_losses = []
    epoch_counts = []
    for k in range(1):  # todo
        val_loss, epoch_count = main(df=df, fold_num=k, idx_shuffled=idx_shuffled)
        val_losses.append(val_loss)
        epoch_counts.append(epoch_count)
    print('===== result =====')
    print(val_losses)
    print(epoch_counts)




