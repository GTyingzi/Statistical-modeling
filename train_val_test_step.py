#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： yingzi
# datetime： 2024/4/22 16:36 
# ide： PyCharm
import torch
from loguru import logger
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

from attack_module.attack import attack_model
from metrics.sequence_label_metric import SeqEntityScore
from metrics.span_metric import GlobalEntityScore, SpanEntityScore
from utils.util import ends_with_crf_or_softmax


def train_base(model, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, args, writer):
    logger.info("Start training...")
    model.train()

    best = 0
    for epoch in range(args.epochs):
        logger.info('start {}-th epoch training'.format(epoch + 1))
        for batch_idx, data in enumerate(tqdm(train_dataloader)):
            step = epoch * len(train_dataloader) + batch_idx + 1
            input_ids = data['input_ids'].to(args.device)
            token_type_ids = data['token_type_ids'].to(args.device)
            attention_mask = data['attention_mask'].to(args.device)

            # 不同模型的输入
            if args.model_class in ['bert-crf', 'bert-bilstm-crf', 'bert-softmax']:
                label_ids = data['label_ids'].to(args.device)
                loss, logits = model(input_ids, attention_mask, token_type_ids, label_ids)
            elif args.model_class == 'bert-globalpointer':
                labels = data['labels'].to(args.device)
                loss, logits = model(input_ids, attention_mask, token_type_ids, labels)
            elif args.model_class == 'bert-span':
                start_label_ids = data['start_label_ids'].to(args.device)
                end_label_ids = data['end_label_ids'].to(args.device)
                loss, start_logits, end_logits = model(input_ids, attention_mask, token_type_ids, labels = (attention_mask, start_label_ids, end_label_ids))

            # 梯度累计
            loss = loss / args.grad_acc_step
            loss.backward()
            # 梯度裁剪
            clip_grad_norm_(model.parameters(), args.max_grad_norm)

            writer.add_scalar('train loss', loss.item(), step)

            # 开启对抗训练
            if args.attack:
                attack_model(args.attack_name, model, input_ids, attention_mask, token_type_ids, label_ids)

            # 进行一定step的梯度累计之后，更新参数
            if step % args.grad_acc_step == 0:
                # 更新参数
                optimizer.step()
                # 更新学习率
                scheduler.step()
                # 清空梯度信息
                optimizer.zero_grad()
            # 评测验证集和测试集上的指标
            if step % args.eval_step == 0:
                best = dev_step(args, model, dev_dataloader, test_dataloader, writer, step, epoch, best)

def evaluate(args, model, dataloader):

    if args.model_class in ["bert-crf", "bert-bilstm-crf", "bert-softmax"]:
        metric = SeqEntityScore(args.id2label, markup= args.markup)
    elif args.model_class == "bert-globalpointer":
        metric = GlobalEntityScore(args.id2label)
    elif args.model_class == "bert-span":
        metric = SpanEntityScore(args.id2label)

    logger.info("***** Running evaluation *****")
    eval_loss = 0.0
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dataloader)):
            input_ids = data['input_ids'].to(args.device)
            token_type_ids = data['token_type_ids'].to(args.device)
            attention_mask = data['attention_mask'].to(args.device)

            if args.model_class in ['bert-crf', 'bert-bilstm-crf', 'bert-softmax']:
                label_ids = data['label_ids'].to(args.device)
                loss, logits = model(input_ids, attention_mask, token_type_ids, label_ids)
            elif args.model_class == 'bert-globalpointer':
                labels = data['labels'].to(args.device)
                loss, logits = model(input_ids, attention_mask, token_type_ids, labels)
            elif args.model_class == 'bert-span':
                start_label_ids = data['start_label_ids'].to(args.device)
                end_label_ids = data['end_label_ids'].to(args.device)
                loss, start_logits, end_logits = model(input_ids, attention_mask, token_type_ids, labels = (attention_mask, start_label_ids, end_label_ids))

            eval_loss += loss.item()
            if ends_with_crf_or_softmax(args.model_class):
                input_lens = (torch.sum(input_ids != 0, dim=-1) - 2).tolist()
                if args.model_class in ['bert-crf', 'bert-bilstm-crf']:
                    preds = model.crf.decode(logits, attention_mask).squeeze(0)
                    preds = preds[:, 1:].tolist()
                else:
                    preds = torch.argmax(logits, dim=2)[:, 1:].tolist()
                label_ids = label_ids[:, 1:].tolist()
                for i in range(len(label_ids)):
                    input_len = input_lens[i]
                    pred = preds[i][:input_len]
                    label = label_ids[i][:input_len]
                    metric.update(pred_paths=[pred], label_paths=[label])
            elif args.model_class == 'bert-globalpointer':
                metric.update(logits, labels)
            elif args.model_class == 'bert-span':
                metric.update(start_end_logits = (start_logits, end_logits), labels = (attention_mask, start_label_ids, end_label_ids))

    logger.info("\n")
    eval_loss = eval_loss / len(dataloader)
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    logger.info("***** Eval results *****")
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    logger.info(info)
    logger.info("***** Entity results *****")
    for key in sorted(entity_info.keys()):
        logger.info("******* %s results ********"%key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        logger.info(info)
    return results

def dev_step(args, model, dev_loader, test_loader, writer, step, epoch,best):
    model.eval()

    logger.info('evaluate dev set')
    dev_result = evaluate(args, model, dev_loader)
    writer.add_scalar('dev loss', dev_result['loss'], step)
    writer.add_scalar('dev f1', dev_result['f1'], step)
    writer.add_scalar('dev precision', dev_result['acc'], step)
    writer.add_scalar('dev recall', dev_result['recall'], step)

    test_result = evaluate(args, model, test_loader)
    writer.add_scalar('test loss', test_result['loss'], step)
    writer.add_scalar('test f1', test_result['f1'], step)
    writer.add_scalar('test precision', test_result['acc'], step)
    writer.add_scalar('test recall', test_result['recall'], step)

    model.train()
    if best < dev_result['f1']:
        best = dev_result['f1']
        logger.info('higher f1 of dev is {} in step {} epoch {}'.format(best, step, epoch + 1))
        logger.info('test is {} in step {} epoch {}'.format(test_result['f1'], step, epoch + 1))
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.output_path)

    return best