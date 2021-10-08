class IQIYIDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.text = df["content"]
        self.char = df["character"]
        self.emo_a = df["emo_a"]
        self.emo_b = df["emo_b"]
        self.emo_c = df["emo_c"]
        self.emo_d = df["emo_d"]
        self.emo_e = df["emo_e"]
        self.emo_f = df["emo_f"]

        self.tokenizer = transformers.BertTokenizer.from_pretrained('inputs/chinese-roberta-wwm-ext',
                                                                    do_lower_case=False)
        self.tokenizer.add_tokens(char_lis)  # todo

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

        text_encoded = self.tokenizer(text,
                                      padding='max_length',
                                      truncation=True,
                                      max_length=my_config.max_len_char)
        if type(char) != type('a'):
            out_pos = 0
        else:
            char_id = self.tokenizer.convert_tokens_to_ids(char)
            out_pos = text_encoded["input_ids"].index(char_id)

        return {
            "input_ids": torch.tensor(text_encoded["input_ids"], dtype=torch.int64),
            "token_type_ids": torch.tensor(text_encoded["token_type_ids"], dtype=torch.int64),
            "attention_mask": torch.tensor(text_encoded["attention_mask"], dtype=torch.int64),
            "out_pos": torch.tensor(out_pos, dtype=torch.int64),
            "emo_a": torch.tensor(emo_a, dtype=torch.int64),
            "emo_b": torch.tensor(emo_b, dtype=torch.int64),
            "emo_c": torch.tensor(emo_c, dtype=torch.int64),
            "emo_d": torch.tensor(emo_d, dtype=torch.int64),
            "emo_e": torch.tensor(emo_e, dtype=torch.int64),
            "emo_f": torch.tensor(emo_f, dtype=torch.int64)
        }
