#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： yingzi
# datetime： 2024/4/21 18:35 
# ide： PyCharm
import json
from os.path import join

import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
import pickle

def ends_with_crf_or_softmax(input_string):
    return input_string.endswith("crf") or input_string.endswith("softmax")

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def write_pickle(x, path):
    with open(path, 'wb') as f:
        pickle.dump(x, f)


def load_lines(path, encoding='utf8'):
    with open(path, 'r', encoding=encoding) as f:
        lines = [line.strip() for line in f.readlines()]
        return lines


def write_lines(lines, path, encoding='utf8'):
    with open(path, 'w', encoding=encoding) as f:
        for line in lines:
            f.writelines('{}\n'.format(line))

def get_entity2idx(data_path, file_name = None):
    if file_name is not None:
        data_path = join(data_path, file_name)
    with open(data_path, 'r', encoding='utf-8') as f:
        value_str = f.read()
    value_dict = json.loads(value_str)

    return value_dict

def load_entity(data_path, file_name = None):
    if file_name is not None:
        data_path = join(data_path, file_name)
    # 读取文本里的每一行，递增为键值对，返回字典
    with open(data_path, 'r', encoding='utf-8') as f:
        value_str = f.read()
        # value_str为字符串，键为实体，值自增获取
        kv = {value: idx for idx, value in enumerate(value_str.split('\n'))}

    return kv

def get_entity_span(entity_span, max_seq_len):
    # entity_span = {'start_offset': 3, 'end_offset': 4, 'label': 'loc', 'span': '北京'}
    # 以start_offset排序
    entity_span = sorted(entity_span, key=lambda x: x['start_offset'])
    # 去掉超过最大长度的实体
    entity_span = [span for span in entity_span if span['start_offset'] < max_seq_len]
    entity_span = [span for span in entity_span if span['end_offset'] < max_seq_len]
    return entity_span

def seed_everything(seed=42):
    """设置整个环境的随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """将序列padding到同一长度"""
    if isinstance(inputs[0], (np.ndarray, list)):
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]

        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode in {'post', 'right'}:
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode in {'pre', 'left'}:
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post/right" or "pre/left".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return np.array(outputs)

    elif isinstance(inputs[0], torch.Tensor):
        assert mode in {'post', 'right'}, '"mode" argument must be "post/right" when element is torch.Tensor'
        if length is not None:
            inputs = [i[:length] for i in inputs]
        return pad_sequence(inputs, padding_value=value, batch_first=True)
    else:
        raise ValueError('"input" argument must be tensor/list/ndarray.')


if __name__ == '__main__':
    data_path = '../datasets/span_data'
    file_name = "entity.txt"
    kv_dict = load_entity(data_path, file_name)