#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： yingzi
# datetime： 2024/4/22 16:07
# ide： PyCharm
import torch
from tqdm import tqdm
from processors.processor import Processor
from processors.data_process import get_text_label_entity
from utils.util import write_pickle, load_pickle, load_lines, sequence_padding, get_entity2idx, get_entity_span
from processors.vocab import Vocabulary
from os.path import join

class BertProcessor(Processor):
    def __init__(self, args, tokenizer):
        super(BertProcessor, self).__init__(args)
        self.Processor_name = "Bert_base"
        self.data_path = args.data_path + args.markup
        self.markup = args.markup
        self.train_file = join(self.data_path, "train.json")
        self.dev_file = join(self.data_path, "dev.json")
        self.test_file = join(self.data_path, "test.json")
        self.max_seq_len = args.max_seq_len - 2
        self.overwrite = args.overwrite
        self.output_path = args.output_path
        self.tokenizer = tokenizer

        # 加载label
        labels = load_lines(join(self.data_path, 'labels.txt'))
        self.label_vocab = Vocabulary(labels, vocab_type='label')

    def get_input_data(self, file):
        examples = get_text_label_entity(file, self.markup)
        features = []
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        o_label_id = self.label_vocab.convert_token_to_id('O')

        for example in tqdm(examples):
            text = example["text"]
            label_ids = example["label"]

            if len(text) > self.max_seq_len:
                text = text[: self.max_seq_len]
                label_ids = label_ids[: self.max_seq_len]

            # 在开头与结尾分别添加[CLS]与[SEP]
            input_ids = [cls_token_id] + self.tokenizer.convert_tokens_to_ids(text) + [sep_token_id]
            label_ids = [o_label_id] + self.label_vocab.convert_tokens_to_ids(label_ids) + [o_label_id]

            input_mask = [1] * len(input_ids)
            token_type_ids = [0] * len(input_ids)
            assert len(input_ids) == len(label_ids)

            feature = {
                'text': text, 'input_ids': input_ids, 'attention_mask': input_mask, 'token_type_ids': token_type_ids,
                'label_ids': label_ids
            }
            features.append(feature)

        return features

    def collate_fn(self,examples):
        batch_text = []
        batch_input_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []
        batch_label_ids = []

        for item in examples:
            batch_text.append(item["text"])
            batch_input_ids.append(item["input_ids"])
            batch_attention_mask.append(item["attention_mask"])
            batch_token_type_ids.append(item["token_type_ids"])
            batch_label_ids.append(item["label_ids"])

        batch_input_ids = torch.tensor(sequence_padding(batch_input_ids,value=self.tokenizer.pad_token_id)).long()
        batch_attention_mask = torch.tensor(sequence_padding(batch_attention_mask)).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()
        batch_label_ids = torch.tensor(sequence_padding(batch_label_ids,value=self.label_vocab.convert_token_to_id('[PAD]'))).long()

        data = {
            "text": batch_text,
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "token_type_ids": batch_token_type_ids,
            "label_ids": batch_label_ids
        }
        return data

class Bert_SpanProcessor(Processor):
    def __init__(self, args, tokenizer):
        super(Bert_SpanProcessor, self).__init__(args)
        self.Processor_name = "Bert_Span"
        self.data_path = args.data_path
        self.markup = args.markup
        self.train_file = join(self.data_path, "train.json")
        self.dev_file = join(self.data_path, "dev.json")
        self.test_file = join(self.data_path, "test.json")
        self.max_seq_len = args.max_seq_len - 2
        self.overwrite = args.overwrite
        self.output_path = args.output_path
        self.tokenizer = tokenizer

        # 加载label
        self.ent2idx = get_entity2idx(args.data_path, file_name='ent2id.json')
        self.label_vocab = Vocabulary(self.ent2idx, 'kv')

    def get_input_data(self, file):
        examples = get_text_label_entity(file, markup=self.markup)
        features = []
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id

        for example in tqdm(examples):
            text = example["text"]
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
                'text': text,
                'input_ids': input_ids,
                'attention_mask': input_mask,
                'token_type_ids': token_type_ids,
                "entity_span": entity_span
            }
            features.append(feature)

        return features

    def collate_fn(self,examples):
        batch_input_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []

        batch_start_label_ids = []
        batch_end_label_ids = []

        for item in examples:
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            token_type_ids = item["token_type_ids"]

            entity_span = item["entity_span"]
            start_label_ids = [0] * len(input_ids)
            end_label_ids = [0] * len(input_ids)
            for entity_item in entity_span:
                start_offset = entity_item["start_offset"]
                end_offset = entity_item["end_offset"]
                label = entity_item["label"]

                start_label_ids[start_offset] = self.ent2idx[label]
                end_label_ids[end_offset] = self.ent2idx[label]

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_start_label_ids.append(start_label_ids)
            batch_end_label_ids.append(end_label_ids)

        batch_input_ids = torch.tensor(sequence_padding(batch_input_ids, value=self.tokenizer.pad_token_id)).long()
        batch_attention_mask = torch.tensor(sequence_padding(batch_attention_mask)).long()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()
        batch_start_label_ids = torch.tensor(sequence_padding(batch_start_label_ids)).long()
        batch_end_label_ids = torch.tensor(sequence_padding(batch_end_label_ids)).long()

        data = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "token_type_ids": batch_token_type_ids,
            "start_label_ids": batch_start_label_ids,
            "end_label_ids": batch_end_label_ids,
        }

        return data