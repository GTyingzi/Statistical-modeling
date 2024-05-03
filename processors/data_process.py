#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： yingzi
# datetime： 2024/4/22 16:15 
# ide： PyCharm
import json

from utils.get_entity import get_entity_bmeso, get_entity_bio

def get_examples_from_seq(data_path, markup, entity2id=None):
    examples = []
    with open(data_path, 'r', encoding='utf-8') as json_file:
        lines = json_file.readlines()

    for line in lines:
        item = json.loads(line)
        sent = item["text"]
        tags = item["label"]
        if markup == "bmeso":
            entity = get_entity_bmeso(tags)
        elif markup == "bio":
            entity = get_entity_bio(tags)

        if entity2id:
            entity = [entity2id[entity_type] for start,end,entity_type in entity]

        assert len(sent) == len(tags)
        example = {
            "text":sent,
            "label":tags,
            "entity":entity
        }
        examples.append(example)
    return examples

def get_examples_from_span(data_path, entity2id=None):
    examples = []
    with open(data_path, 'r', encoding='utf-8') as json_file:
        lines = json_file.readlines()

    for line in lines:
        item = json.loads(line)
        sent = item["text"]
        entity = item["entities"]
        if entity2id:
            entity = [entity2id[entity_type] for start, end, entity_type, content in entity]

        example = {
            "text":sent,
            "label":None,
            "entity":entity
        }
        examples.append(example)
    return examples

def get_text_label_entity(data_path, markup = "bmeso", entity2id=None):
    if markup in ["bmeso", "bio"]:
        return get_examples_from_seq(data_path, markup, entity2id)
    elif markup == "span":
        return get_examples_from_span(data_path, entity2id)

    return None