# model_v2考虑上下文
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

# 结合上文
def create_context(df_nodup, context_len = 128):
    context_dic = {df_nodup['content'][0]:df_nodup['content'][0]}
    for idx, text in enumerate(df_nodup['content'][1:], 1):
        if text not in context_dic:
            context_dic[text] = text
            text_len = len(text)
            pre_idx = idx-1
            while pre_idx >= 0:
                pre_text = df_nodup['content'][pre_idx]
                pre_len = len(pre_text)
                if text_len + pre_len <= context_len:
                    context_dic[text] = pre_text + context_dic[text]
                    text_len += pre_len
                    pre_idx -= 1
                else:
                    break
    return context_dic

def write_pickle(file, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(file, f)

context_train_dic = create_context(train_df_nodup)
context_test_dic = create_context(test_df_nodup)
# write_pickle(context_train_dic, './outputs/context_train_dic')
# write_pickle(context_test_dic, './outputs/context_test_dic')


# 删除情感为空的样本
drop_idx_train = [i for i, j in enumerate(train_df['emotions']) if not isinstance(j, str)]
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
    def __init__(self, n_folds=5, n_epochs=7, batch_size=16, batch_size_accu=256, patience=3,
                 lr=2e-5, max_len_char=410, ways_of_mask=2):
        self.n_folds = n_folds
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.batch_size_accu = batch_size_accu
        self.patience = patience
        self.lr = lr  # 学习率参考https://github.com/ymcui/Chinese-BERT-wwm
        self.max_len_char = max_len_char  # (412-4)+2
        self.ways_of_mask = ways_of_mask  # dynamic masking
        self.tokenizer = transformers.BertTokenizer.from_pretrained('inputs/chinese-roberta-wwm-ext',
                                                                    do_lower_case=False)
        self.tokenizer.add_tokens(char_lis)  # todo


my_config = Config()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, context_dic):
        self.text = df['content']
        self.char = df["character"]
        self.emo_a = df["emo_a"]
        self.emo_b = df["emo_b"]
        self.emo_c = df["emo_c"]
        self.emo_d = df["emo_d"]
        self.emo_e = df["emo_e"]
        self.emo_f = df["emo_f"]
        self.tokenizer = my_config.tokenizer
        self.context_dic = context_dic

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = self.text[item]
        context_text = self.context_dic[text]
        char = self.char[item]
        emo_a = self.emo_a[item]
        emo_b = self.emo_b[item]
        emo_c = self.emo_c[item]
        emo_d = self.emo_d[item]
        emo_e = self.emo_e[item]
        emo_f = self.emo_f[item]

        encoded_inputs = self.tokenizer(context_text,
                                        padding='max_length',
                                        truncation=True,
                                        max_length=my_config.max_len_char)
        if isinstance(char, str):
            char_id = self.tokenizer.convert_tokens_to_ids(char)
            try:
                out_pos = encoded_inputs["input_ids"].index(char_id)
                # todo model_v3
                # inputids = self.tokenizer(text)["input_ids"]
                # out_pos = inputids.index(char_id) - len(inputids) + sum(encoded_inputs["attention_mask"])
            except:
                out_pos = 0
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


class ModelClf(transformers.BertPreTrainedModel):
    '''为每一种情感训练一个分类器'''

    def __init__(self, config):
        super(ModelClf, self).__init__(config)
        self.bert_mlm = transformers.BertForMaskedLM.from_pretrained('inputs/chinese-roberta-wwm-ext',
                                                                     config=config)  # 哈工大预训练模型
        self.bert_mlm.resize_token_embeddings(len(my_config.tokenizer))  # todo word_embedding.shape=(21151,768)
        self.bert_mlm.load_state_dict(torch.load('./outputs/mlm_checkpoint0_nonaccu.pt'))
        self.la = torch.nn.Linear(768, 4)
        self.lb = torch.nn.Linear(768, 4)
        self.lc = torch.nn.Linear(768, 4)
        self.ld = torch.nn.Linear(768, 4)
        self.le = torch.nn.Linear(768, 4)
        self.lf = torch.nn.Linear(768, 4)

    def forward(self, input_ids, attention_mask, token_type_ids, out_pos):
        out = self.bert_mlm(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        sample_id = torch.tensor(np.arange(input_ids.shape[0]), dtype=torch.int64)
        out = out.hidden_states[-1][sample_id, out_pos, :]  # shape=(batch_size, hidden_size)
        outa = self.la(out)
        outb = self.lb(out)
        outc = self.lc(out)
        outd = self.ld(out)
        oute = self.le(out)
        outf = self.lf(out)
        return outa, outb, outc, outd, oute, outf


'''training：定义一个epoch上的训练'''


def training(model, dataloader, loss_fn, optimizer, scheduler, device):
    model.train()
    dataloader_tqdm = tqdm.tqdm(dataloader)
    losses = 0
    for data in dataloader_tqdm:
        outputs = model(input_ids=data['input_ids'].to(device=device),
                        attention_mask=data['attention_mask'].to(device=device),
                        token_type_ids=data['token_type_ids'].to(device=device),
                        out_pos=data['out_pos'].to(device=device))
        lossa = loss_fn(outputs[0], data['emo_a'].to(device=device))
        lossb = loss_fn(outputs[1], data['emo_b'].to(device=device))
        lossc = loss_fn(outputs[2], data['emo_c'].to(device=device))
        lossd = loss_fn(outputs[3], data['emo_d'].to(device=device))
        losse = loss_fn(outputs[4], data['emo_e'].to(device=device))
        lossf = loss_fn(outputs[5], data['emo_f'].to(device=device))
        loss = lossa + lossb + lossc + lossd + losse + lossf
        losses += loss.item()
        optimizer.zero_grad()
        loss.backward()  # 可以释放计算图
        optimizer.step()
        # scheduler.step()
        dataloader_tqdm.set_postfix({'loss': loss.item()})  # 当前batch上平均每个样本的loss
    return losses / len(dataloader)  # 一个epoch上平均每个样本的损失


'''evaluating：定义一个epoch上的evaluating'''


def evaluating(model, dataloader, loss_fn, device):
    model.eval()
    with torch.no_grad():
        dataloader_tqdm = tqdm.tqdm(dataloader)
        losses_val = 0
        for data in dataloader_tqdm:
            outputs = model(input_ids=data['input_ids'].to(device=device),
                            attention_mask=data['attention_mask'].to(device=device),
                            token_type_ids=data['token_type_ids'].to(device=device),
                            out_pos=data['out_pos'].to(device=device))
            lossa = loss_fn(outputs[0], data['emo_a'].to(device=device))
            lossb = loss_fn(outputs[1], data['emo_b'].to(device=device))
            lossc = loss_fn(outputs[2], data['emo_c'].to(device=device))
            lossd = loss_fn(outputs[3], data['emo_d'].to(device=device))
            losse = loss_fn(outputs[4], data['emo_e'].to(device=device))
            lossf = loss_fn(outputs[5], data['emo_f'].to(device=device))
            loss = lossa + lossb + lossc + lossd + losse + lossf
            losses_val += loss.item()
            dataloader_tqdm.set_postfix({'loss': loss.item()})
    return losses_val / len(dataloader)


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
    model_config = transformers.BertConfig.from_pretrained('inputs/chinese-roberta-wwm-ext/config.json')
    model_config.output_hidden_states = True
    model = ModelClf(config=model_config)
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
                                   path='./outputs/modelv1_checkpoint%d.pt' % fold_num)
    loss_fn = nn.CrossEntropyLoss()

    epoch_record = None
    for epoch in range(1, my_config.n_epochs + 1):
        epoch_record = epoch
        training(model, train_dataloader, loss_fn, optimizer, scheduler, device)
        loss_val = evaluating(model, val_dataloader, loss_fn, device)
        early_stopping(loss_val, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    if early_stopping.early_stop:
        return early_stopping.val_loss_min, epoch_record - my_config.patience
    else:
        return early_stopping.val_loss_min, my_config.n_epochs


'''predict'''


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.text = df['content']
        self.char = df["character"]
        self.tokenizer = my_config.tokenizer

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = self.text[item]
        char = self.char[item]

        encoded_inputs = self.tokenizer(text,
                                        padding='max_length',
                                        truncation=True,
                                        max_length=my_config.max_len_char)
        if isinstance(char, str):
            char_id = self.tokenizer.convert_tokens_to_ids(char)
            try:
                out_pos = encoded_inputs["input_ids"].index(char_id)
            except:
                out_pos = 0
        else:
            out_pos = 0

        return {
            "input_ids": torch.tensor(encoded_inputs["input_ids"], dtype=torch.int64),
            "token_type_ids": torch.tensor(encoded_inputs["token_type_ids"], dtype=torch.int64),
            "attention_mask": torch.tensor(encoded_inputs["attention_mask"], dtype=torch.int64),
            "out_pos": torch.tensor(out_pos, dtype=torch.int64)
        }


def predicting(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        results = []
        dataloader_tqdm = tqdm.tqdm(dataloader)
        for data in dataloader_tqdm:
            outputs = model(input_ids=data['input_ids'].to(device=device),
                            attention_mask=data['attention_mask'].to(device=device),
                            token_type_ids=data['token_type_ids'].to(device=device),
                            out_pos=data['out_pos'].to(device=device))
            outputa, outputb, outputc, outputd, outpute, outputf = outputs
            _, ra = torch.max(outputa, dim=1)
            _, rb = torch.max(outputb, dim=1)
            _, rc = torch.max(outputc, dim=1)
            _, rd = torch.max(outputd, dim=1)
            _, re = torch.max(outpute, dim=1)
            _, rf = torch.max(outputf, dim=1)
            r = torch.stack([ra, rb, rc, rd, re, rf], dim=0)
            r = r.T
            results.append(r)
        results = torch.cat(results, dim=0)
        return results


def predict_result(df, fold_num):
    ########## DataLoader ##########
    test_dataset = TestDataset(df)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=my_config.batch_size,
                                                  shuffle=False)

    ########## model ##########
    model_config = transformers.BertConfig.from_pretrained('inputs/chinese-roberta-wwm-ext/config.json')
    model_config.output_hidden_states = True
    model = ModelClf(config=model_config)
    model.load_state_dict(torch.load('./outputs/modelv1_checkpoint%d.pt' % fold_num))  # , map_location=torch.device('cpu')
    model = model.to(device=device)

    results = predicting(model, test_dataloader, device)
    results = results.cpu().numpy()
    results = results.astype(str)
    emotion = [','.join(i) for i in results]
    test_df['emotion%d' % fold_num] = emotion
    submission = test_df.loc[:, ['id', 'emotion%d' % fold_num]]
    submission.to_csv('./outputs/submission%d.csv' % fold_num, sep='\t', header=True, index=False)


if __name__ == '__main__':
    df = train_df
    idx_shuffled = np.random.permutation(df.shape[0])  # 保证只打乱一次下标

    val_losses = []
    epoch_counts = []
    for k in range(1):  # todo my_config.n_folds
        val_loss, epoch_count = main(df=df, fold_num=k, idx_shuffled=idx_shuffled)
        val_losses.append(val_loss)
        epoch_counts.append(epoch_count)

        predict_result(test_df, k)  # 预测

    print('===== train_result =====')
    print(val_losses)
    print(epoch_counts)


