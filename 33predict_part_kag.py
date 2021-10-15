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
            out_pos = encoded_inputs["input_ids"].index(char_id)
        else:
            out_pos = 0


        return {
            "input_ids": torch.tensor(encoded_inputs["input_ids"], dtype=torch.int64),
            "token_type_ids": torch.tensor(encoded_inputs["token_type_ids"], dtype=torch.int64),
            "attention_mask": torch.tensor(encoded_inputs["attention_mask"], dtype=torch.int64),
            "out_pos": torch.tensor(out_pos, dtype=torch.int64)
        }


class ModelClfTest(transformers.BertPreTrainedModel):
    '''为每一种情感训练一个分类器'''
    def __init__(self, config):
        super(ModelClfTest, self).__init__(config)
        self.bert_mlm = transformers.BertForMaskedLM.from_pretrained('../input/hflchineserobertawwmext',
                                                                     config=config)  # 哈工大预训练模型
        self.bert_mlm.resize_token_embeddings(len(my_config.tokenizer))  # todo word_embedding.shape=(21151,768)
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


def predict(test_df, fold_num):
    ########## DataLoader ##########
    test_dataset = TestDataset(test_df)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=my_config.batch_size,
                                                 shuffle=False)

    ########## model ##########
    model_config = transformers.BertConfig.from_pretrained('../input/hflchineserobertawwmext/config.json')
    model_config.output_hidden_states = True
    model = ModelClfTest(config=model_config)
    model.load_state_dict(torch.load('./outputs/modelv1_checkpoint%d.pt' % fold_num))  # , map_location=torch.device('cpu')
    model = model.to(device=device)

    ########## predict ##########
    model.eval()
    with torch.no_grad():
        results = []
        for data in tqdm.tqdm(test_dataloader):
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
        results = results.cpu().numpy()
    return results

if __name__ == '__main__':
    for k in range(1):  # todo my_config.n_folds
        results = predict(test_df=test_df, fold_num=k)
        results = results.astype(str)
        emotion = [','.join(i) for i in results]
        test_df['emotion'] = emotion
        submission = test_df.loc[:, ['id', 'emotion']]
        submission.to_csv('./outputs/submission.csv', sep='\t', header=True, index=False)



