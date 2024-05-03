#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： yingzi
# datetime： 2024/4/22 16:09 
# ide： PyCharm

import os
from os.path import join
from loguru import logger
from utils.util import write_pickle, load_pickle
from os.path import join
from torch.utils.data import Dataset

class Processor(object):

    def __init__(self, config):
        self.data_path = config.data_path
        self.overwrite = config.overwrite
        self.train_file = join(config.data_path, "train.json")
        self.dev_file = join(config.data_path, "dev.json")
        self.test_file = join(config.data_path, "test.json")

    def get_input_data(self, file):
        raise NotImplemented()

    def get_train_data(self):
        logger.info('loading train data')
        save_path = join(self.data_path, self.Processor_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        file_name = self.Processor_name + "_train.pkl"
        file_path = join(save_path, file_name)
        if self.overwrite or not os.path.exists(file_path):
            features = self.get_input_data(self.train_file)
            write_pickle(features, file_path)
        else:
            features = load_pickle(file_path)
        train_dataset = NERDataset(features)
        logger.info('len of train data:{}'.format(len(features)))
        return train_dataset

    def get_dev_data(self):
        logger.info('loading dev data')
        save_path = join(self.data_path, self.Processor_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        file_name = self.Processor_name + "_val.pkl"
        file_path = join(save_path, file_name)
        if self.overwrite or not os.path.exists(file_path):
            features = self.get_input_data(self.dev_file)
            write_pickle(features, file_path)
        else:
            features = load_pickle(file_path)
        dev_dataset = NERDataset(features)
        logger.info('len of dev data:{}'.format(len(features)))
        return dev_dataset

    def get_test_data(self):
        logger.info('loading test data')
        save_path = join(self.data_path, self.Processor_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        file_name = self.Processor_name + "_test.pkl"
        file_path = join(save_path, file_name)
        if self.overwrite or not os.path.exists(file_path):
            features = self.get_input_data(self.test_file)
            write_pickle(features, file_path)
        else:
            features = load_pickle(file_path)
        test_dataset = NERDataset(features)
        logger.info('len of test data:{}'.format(len(features)))
        return test_dataset


class NERDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]
        return feature