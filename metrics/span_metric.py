#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： yingzi
# datetime： 2024/4/24 16:45 
# ide： PyCharm
from collections import Counter
import numpy as np
import torch


class GlobalEntityScore(object):
    def __init__(self,id2label,threshold=0):
        self.id2label = id2label
        self.threshold = threshold
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([x[2] for x in self.origins])
        found_counter = Counter([x[2] for x in self.founds])
        right_counter = Counter([x[2] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'acc': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self,scores,labels):
        # 实体粒度
        for score,label in zip(scores, labels):
            pre_entities = self.decode(score.cpu().numpy())
            label_entities = self.decode(label.cpu().numpy())
            self.origins.extend(label_entities)
            self.founds.extend(pre_entities)
            self.rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])

    def decode(self, score):
        R = set()

        for entity, start, end in zip(*np.where(score > self.threshold)):
            R.add((start, end, self.id2label[entity]))

        return R


class SpanEntityScore(object):
    def __init__(self, id2label):
        self.origins = []
        self.founds = []
        self.rights = []
        self.id2label = id2label

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([x[-1] for x in self.origins])
        found_counter = Counter([x[-1] for x in self.founds])
        right_counter = Counter([x[-1] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'acc': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, start_end_logits, labels):
        start_logit, end_logit = start_end_logits
        mask, start_ids, end_ids = labels

        pre_entities = self.decode(start_logit, end_logit, mask)
        label_entities = self.decode(start_ids, end_ids)
        self.founds.extend(pre_entities)
        self.origins.extend(label_entities)
        self.rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])


    def decode(self, start_preds, end_preds, mask=None):
        '''返回实体的start, end
        '''
        predict_entities = set()
        if mask is not None:  # 预测的把query和padding部分mask掉
            start_preds = torch.argmax(start_preds, -1) * mask
            end_preds = torch.argmax(end_preds, -1) * mask

        start_preds = start_preds.cpu().numpy()
        end_preds = end_preds.cpu().numpy()

        for bt_i in range(start_preds.shape[0]):
            start_pred = start_preds[bt_i]
            end_pred = end_preds[bt_i]
            # 统计每个样本的结果
            for i, s_type in enumerate(start_pred):
                if s_type == 0:
                    continue
                for j, e_type in enumerate(end_pred[i:]):
                    if s_type == e_type:
                        # [样本id, 实体起点，实体终点，实体类型]
                        predict_entities.add((bt_i, i, i + j, self.id2label[s_type]))
                        break
        return predict_entities

