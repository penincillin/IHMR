import os, sys, shutil
import os.path as osp
import random
from datetime import datetime
import torch
from data.baseline_dataset import BaselineDataset
from data.opt_dataset import OPTDataset
from data.mlp_dataset import MLPDataset
import torch.utils.data as data
import numpy as np


class ComposeDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt

        if opt.model_type == "baseline":
            dataset_info = dict(
                hand26m = ("hand26m", opt.hand26m_anno_path, 'hand26m/image'),
            )
        elif opt.model_type == "opt":
            dataset_info = dict(
                hand26m = ("hand26m", opt.hand26m_anno_path, opt.hand26m_pred_path, 'hand26m/image'),
            )
        else:
            assert opt.model_type == "mlp"
            dataset_info = dict(
                hand26m = ("hand26m", opt.hand26m_anno_path, opt.hand26m_pred_path, 'hand26m/image'),
            )
        
        Dataset = dict(
            baseline = BaselineDataset,
            opt = OPTDataset,
            mlp = MLPDataset,
        )[opt.model_type]

        all_potential_datasets = dict()
        for dataset_name in dataset_info:
            dataset = Dataset(opt, dataset_info[dataset_name])
            all_potential_datasets[dataset_name] = dataset
        
        candidate_datasets = list()
        if opt.model_type in ["baseline", "mlp"]:
            if opt.isTrain:
                datasets_str = opt.train_datasets
                train_dataset_names = datasets_str.strip().split(',')
                for dataset_name in train_dataset_names:
                    candidate_datasets.append(all_potential_datasets[dataset_name])
            else:
                candidate_datasets.append(all_potential_datasets[opt.test_dataset])
        else:
            assert opt.model_type == "opt"
            candidate_datasets.append(all_potential_datasets[opt.opt_dataset])
        
        assert(len(candidate_datasets)>0)
        self.all_datasets = candidate_datasets

        for dataset in candidate_datasets:
            dataset.load_data()

        if opt.process_rank <= 0: 
            for dataset in candidate_datasets:
                print('{} dataset has {} data'.format(dataset.name, len(dataset)-dataset.num_add))

        self.set_index_map()


    def set_index_map(self):
        total_data_num = self.__len__()
        index_map = list()
        for dataset_id, dataset in enumerate(self.all_datasets):
            dataset_len = len(dataset)
            index_map += [(dataset_id, idx) for idx in range(dataset_len)]
        self.index_map = index_map


    def __getitem__(self, index):
        dataset_id, dataset_index = self.index_map[index]
        dataset = self.all_datasets[dataset_id]
        data = dataset.getitem(dataset_index)
        return data


    def shuffle_data(self):
        for dataset in self.all_datasets:
            random.shuffle(dataset.data_list)
    

    def __len__(self):
        total_data_num = sum([len(dataset) for dataset in self.all_datasets])
        return total_data_num

    @property
    def name(self):
        'ComposeDataset'