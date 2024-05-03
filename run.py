#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： yingzi
# datetime： 2024/4/21 18:32 
# ide： PyCharm
import os
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
import time

from models.sequence_label_model import Bert_Bilstm_Crf, Bert_Crf, Bert_Softmax
from models.span_model import Bert_GlobalPointer, Bert_Span
from processors.global_process import Bert_GPPocessor
from train_val_test_step import train_base, evaluate
from utils.options import Args
from utils.trainer import get_optimizer
from utils.util import seed_everything, ends_with_crf_or_softmax
from transformers import BertTokenizer, BertConfig
import torch
from torch.utils.data import DataLoader
from os.path import join
from processors.base_processor import BertProcessor, Bert_SpanProcessor

PROCESSOR_CLASS = {
    'bert-crf': BertProcessor,
    'bert-bilstm-crf': BertProcessor,
    'bert-softmax': BertProcessor,
    'bert-globalpointer': Bert_GPPocessor,
    'bert-span': Bert_SpanProcessor,
}

MODEL_CLASS = {
    'bert-crf': Bert_Crf,
    'bert-bilstm-crf': Bert_Bilstm_Crf,
    'bert-softmax': Bert_Softmax,
    'bert-globalpointer': Bert_GlobalPointer,
    'bert-span': Bert_Span,
}

def main(args, writer):
    # 分词器
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)
    # 数据处理器
    processor = PROCESSOR_CLASS[args.model_class](args, tokenizer)
    args.id2label = processor.label_vocab.idx2token
    if ends_with_crf_or_softmax(args.model_class):
        args.ignore_index = processor.label_vocab.convert_token_to_id('[PAD]')

    # 初始化模型
    config = BertConfig.from_pretrained(args.pretrain_model_path)
    config.num_labels = processor.label_vocab.size
    config.loss_type = args.loss_type
    if args.model_class == 'bert-globalpointer':
        config.inner_dim = args.inner_dim
        config.RoPE = args.RoPE
    model = MODEL_CLASS[args.model_class].from_pretrained(args.pretrain_model_path, config=config).to(args.device)

    if args.do_train:
        # 加载数据集
        train_dataset = processor.get_train_data()
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True,
                                      num_workers=args.num_workers,collate_fn=processor.collate_fn)
        dev_dataset = processor.get_dev_data()
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size_eval, shuffle=False,
                                    num_workers=args.num_workers,collate_fn=processor.collate_fn)
        test_dataset = processor.get_test_data()
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_eval, shuffle=False,
                                     num_workers=args.num_workers,collate_fn=processor.collate_fn)
        t_total = len(train_dataloader) // args.grad_acc_step * args.epochs
        warmup_steps = int(t_total * args.warmup_proportion)
        optimizer, scheduler = get_optimizer(model, args, warmup_steps, t_total)
        train_base(model, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, args, writer)

    if args.do_eval:
        test_dataset = processor.get_test_data()
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_eval, shuffle=False,
                                     num_workers=args.num_workers,collate_fn=processor.collate_fn)
        model = MODEL_CLASS[args.model_class].from_pretrained(args.output_path, config=config).to(args.device)
        model.eval()

        result = evaluate(args, model, test_dataloader)
        logger.info(
            'testset precision:{}, recall:{}, f1:{}, loss:{}'.format(result['acc'], result['recall'], result['f1'],
                                                                     result['loss']))

if __name__ == '__main__':
    args = Args().get_parser()
    seed_everything(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else "cpu")

    # 设置输出路径
    args.output_path = join(args.output_path, args.model_class)
    if args.attack:
        args.output_path = args.output_path + '_' + args.attack_name

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if args.do_train:
        cur_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        logger.add(join(args.output_path, 'train-{}.log'.format(cur_time)))
        logger.info(args)
        writer = SummaryWriter(args.output_path)
    main(args, writer)