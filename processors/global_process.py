#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： yingzi
# datetime： 2024/4/24 16:25 
# ide： PyCharm
from os.path import join

import numpy as np
import torch
from tqdm import tqdm

from processors.data_process import get_text_label_entity
from processors.processor import Processor
from processors.vocab import Vocabulary
from utils.util import get_entity_span, sequence_padding, load_entity


class Bert_GPPocessor(Processor):
    def __init__(self, args, tokenizer):
        super(Bert_GPPocessor, self).__init__(args)
        self.Processor_name = "Bert_GP"
        self.data_path = args.data_path
        self.markup = args.markup
        self.train_file = join(args.data_path, "train.json")
        self.dev_file = join(args.data_path, "dev.json")
        self.test_file = join(args.data_path, "test.json")
        self.max_seq_len = args.max_seq_len - 2
        self.overwrite = args.overwrite
        self.output_path = args.output_path
        self.tokenizer = tokenizer
        self.ent2idx = load_entity(args.data_path, file_name='entity.txt')
        self.label_vocab = Vocabulary(self.ent2idx, 'kv')

    def get_input_data(self,file):
        examples = get_text_label_entity(file, markup=self.markup)
        features = []
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id

        for example in tqdm(examples):
            text = example['text']
            entity_span = example["entity"]

            text_len = len(text)

            if text_len > self.max_seq_len:
                text = text[:self.max_seq_len]

                entity_span = get_entity_span(entity_span, self.max_seq_len)

            # 在开头与结尾分别添加[CLS]与[SEP]
            input_ids = [cls_token_id] + self.tokenizer.convert_tokens_to_ids(text) + [sep_token_id]

            input_mask = [1] * len(input_ids)
            token_type_ids = [0] * len(input_ids)

            feature = {
                "input_ids": input_ids,
                "attention_mask": input_mask,
                "token_type_ids": token_type_ids,
                "entity_span": entity_span
            }
            features.append(feature)

        return features

    def collate_fn(self,examples):
        batch_input_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []
        batch_labels = []

        for item in examples:
            batch_input_ids.append(item["input_ids"])
            batch_attention_mask.append(item["attention_mask"])
            batch_token_type_ids.append(item["token_type_ids"])

            labels = np.zeros((len(self.ent2idx), self.max_seq_len + 2 , self.max_seq_len + 2))
            entity_span = item["entity_span"]
            # entity_item = {'start_offset': 3, 'end_offset': 4, 'label': 'loc', 'span': '北京'}
            for entity_item in entity_span:
                start_offset = entity_item["start_offset"]
                end_offset = entity_item["end_offset"]
                label = entity_item["label"]
                labels[self.ent2idx[label], start_offset, end_offset] = 1
            batch_labels.append(labels[:, :len(item["input_ids"]), :len(item["input_ids"])])

        batch_input_ids = torch.tensor(sequence_padding(batch_input_ids,value=self.tokenizer.pad_token_id)).long()
        batch_attention_mask = torch.tensor(sequence_padding(batch_attention_mask)).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()
        batch_labels = torch.tensor(sequence_padding(batch_labels, seq_dims=3)).long()

        data = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "token_type_ids": batch_token_type_ids,
            'labels': batch_labels
        }
        return data