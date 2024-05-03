#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： yingzi
# datetime： 2024/4/22 16:20 
# ide： PyCharm
def get_entity_bio(seq, id2label=None):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
        id2label (dict): a dictionary mapping label ids to label strings.
    Returns:
        list: list of [start, end, 实体] lists.
    Example:
        # >>> seq_data = ['B-name', 'I-name', 'I-name', 'O', 'B-loc', 'B-loc', 'I-loc', 'O', 'B-org', 'I-org', 'I-org', 'I-org']
        # >>> get_entity_bio(seq_data)
        # [[0, 2, 'name'], [4, 5, 'loc'], [8, 11, 'org']]
    """
    entities = []
    entity_start = None
    entity_type = None

    for idx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith('B-'):
            if entity_start is not None:
                entities.append([entity_start, idx - 1, entity_type])
            entity_start = idx
            entity_type = tag[2:]
        elif tag.startswith('I-'):
            if entity_start is None:
                continue  # Invalid sequence, ignore
        elif tag.startswith('O'):
            if entity_start is not None:
                entities.append([entity_start, idx - 1, entity_type])
                entity_start = None
                entity_type = None

    if entity_start is not None:
        entities.append([entity_start, len(seq) - 1, entity_type])

    return entities


def get_entity_bmeso(seq, id2label=None):
    """Gets entities from sequence.
    note: BMESO
    Args:
        seq (list): sequence of labels.
        id2label (dict): a dictionary mapping label ids to label strings.
    Returns:
        list: list of [start, end, 实体] lists.
    Example:
        # >>> seq_data = ['B-name', 'M-name', 'E-name', 'O', 'S-loc', 'B-loc', 'E-loc', 'O', 'B-org', 'M-org', 'M-org', 'E-org']
        # >>> get_entity_bmeso(seq_data)
        # [[0, 2, 'name'], [4, 4, 'loc'], [5, 6, 'loc'], [8, 11, 'org']]
    """
    entities = []
    entity_start = None
    entity_type = None

    for idx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith('B-'):
            entity_start = idx
            entity_type = tag[2:]
        elif tag.startswith('M-'):
            if entity_start is None:
                continue  # Invalid sequence, ignore
        elif tag.startswith('E-'):
            if entity_start is None:
                continue  # Invalid sequence, ignore
            entities.append([entity_start, idx, entity_type])
            entity_start = None
            entity_type = None
        elif tag.startswith('S-'):
            entities.append([idx, idx, tag[2:]])

    return entities



def get_entities(seq, id2label, markup='bios'):
    '''
    :param seq:
    :param id2label:
    :param markup:
    :return:
    '''
    assert markup in ['seq_data','bios','bmeso']
    if markup == 'seq_data':
        return get_entity_bio(seq,id2label)
    elif markup == 'bmeso':
        return get_entity_bmeso(seq,id2label)