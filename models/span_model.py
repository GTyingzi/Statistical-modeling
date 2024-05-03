#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： yingzi
# datetime： 2024/4/22 14:23 
# ide： PyCharm
import torch.nn as nn
import torch

from transformers import BertModel, BertPreTrainedModel
from losses.mcc_loss import MultilabelCategoricalCrossentropy


class Bert_Span(BertPreTrainedModel):
    def __init__(self, config):
        super(Bert_Span, self).__init__(config)
        self.bert = BertModel(config)
        self.mid_linear = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.num_labels = config.num_labels
        self.start_fc = nn.Linear(config.hidden_size, self.num_labels + 1)
        self.end_fc = nn.Linear(config.hidden_size, self.num_labels + 1)
        self.loss_type = config.loss_type

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        context_outputs = self.bert(input_ids, attention_mask, token_type_ids)
        last_hidden_state = context_outputs[0]

        mid_output = self.mid_linear(last_hidden_state)
        start_logits = self.start_fc(mid_output)
        end_logits = self.end_fc(mid_output)
        outputs = (start_logits, end_logits)

        if labels:
            loss = self.get_loss((start_logits, end_logits), labels)
            outputs = (loss,) + outputs

        return outputs

    def get_loss(self, outputs, labels):
        start_logits, end_logits = outputs
        mask, start_labels, end_labels = labels

        start_logits = start_logits.view(-1, self.num_labels + 1)
        end_logits = end_logits.view(-1, self.num_labels + 1)

        # 去掉 padding 部分的标签，计算真实 loss
        active_loss = mask.view(-1) == 1
        active_start_logits = start_logits[active_loss]
        active_end_logits = end_logits[active_loss]
        active_start_labels = start_labels.view(-1)[active_loss]
        active_end_labels = end_labels.view(-1)[active_loss]

        start_loss = nn.CrossEntropyLoss()(active_start_logits, active_start_labels)
        end_loss = nn.CrossEntropyLoss()(active_end_logits, active_end_labels)
        loss = start_loss + end_loss

        return loss


class Bert_GlobalPointer(BertPreTrainedModel):
    def __init__(self, config):
        super(Bert_GlobalPointer, self).__init__(config)
        self.bert = BertModel(config)
        self.ent_type_size = config.num_labels
        self.inner_dim = config.inner_dim
        self.hidden_size = config.hidden_size
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)

        self.RoPE = config.RoPE

        self.loss_type = config.loss_type


    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        context_outputs = self.bert(input_ids, attention_mask, token_type_ids)
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        # outputs:(batch_size, seq_len, ent_type_size*inner_dim*2)
        outputs = self.dense(last_hidden_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)  # 把ent_type_size给抽离出来了
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]

        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = (logits - mask * 1e12) / self.inner_dim ** 0.5

        if labels is not None:
            assert self.loss_type in ['mcc']
            if self.loss_type == 'mcc':
                loss_mcc = MultilabelCategoricalCrossentropy()

            loss = loss_mcc(logits, labels)
            logits = (loss,) + (logits,)

        return logits