#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： yingzi
# datetime： 2024/4/22 15:20 
# ide： PyCharm
import argparse

class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser: argparse.ArgumentParser):
        parser.add_argument("--device", type=str, default='gpu', choices=['gpu', 'cpu'], help="gpu or cpu")
        parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行训练')
        parser.add_argument("--output_path", type=str, default='output/', help='模型与预处理数据的存放位置')
        parser.add_argument("--do_train", action='store_true', default=True)
        parser.add_argument("--do_eval", action='store_true', default=True)
        parser.add_argument("--overwrite", action='store_true', default=False, help="覆盖数据处理的结果")

        parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
        parser.add_argument('--eps', default=1.0e-08, type=float, required=False, help='AdamW优化器的衰减率')
        parser.add_argument("--epochs", type=int, default=5)
        parser.add_argument("--batch_size_train", type=int, default=4)
        parser.add_argument("--batch_size_eval", type=int, default=256)
        parser.add_argument("--eval_step", type=int, default=50, help="训练多少步，查看验证集的指标")
        parser.add_argument("--max_seq_len", type=int, default=512, help="输入的最大长度")

        parser.add_argument('--grad_acc_step', default=1, type=int, required=False, help='梯度积累的步数')
        parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False, help='梯度裁剪阈值')
        parser.add_argument('--seed', type=int, default=42, help='设置随机种子')
        parser.add_argument('--num_workers', type=int, default=0, help="dataloader加载数据时使用的线程数量")
        parser.add_argument('--warmup_proportion', type=float, default=0.1, help='Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.')

        parser.add_argument("--attack", action='store_true', default=False, help="开启对抗训练")
        parser.add_argument("--attack_name", type=str, choices=['fgm', 'pgd', 'fgsm'], default='fgm',
                            help="对抗训练的方法")

        parser.add_argument("--pretrain_model_path", type=str, default="../bert-base-chinese")
        parser.add_argument("--lr", type=float, default=1e-5, help='Bert模块的学习率')
        parser.add_argument('--other_lr', default=2e-5, type=float,
                            help='除Bert以外模块的学习率，如LSTM、CRF等')

        parser.add_argument("--data_path", type=str, default="datasets/span_data/", help='数据集存放路径')
        parser.add_argument("--model_class", type=str,
                            choices=['bert-softmax', 'bert-crf', 'bert-bilstm-crf', 'bert-globalpointer', 'bert-span'],
                            default='bert-globalpointer', help='模型类别')

        # 模型参数配置
        parser.add_argument('--markup', default='span', type=str, choices=['bio', 'bmeso', 'span'], help='数据集的标注方式')
        parser.add_argument('--loss_type', default='mcc', type=str, choices=['mcc'],
                            help='损失函数类型')
        parser.add_argument("--inner_dim", type=int, default=64, help="内部维度")
        parser.add_argument("--RoPE", action='store_true', default=True, help="是否使用RoPE")

        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        return parser.parse_args()