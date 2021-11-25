import os
import os.path as osp
import sys
import shutil
import time
from datetime import datetime
import torch
import numpy
import random
from torch.multiprocessing import Process
import torch.distributed as dist
import torch.multiprocessing as mp
import cv2
import numpy as np
import pdb
import ry_utils
from options.opt_options import OptOptions
from utils.init_utils import init_dist, init_opt
from utils.opt_utils import TimeStat
from data.data_loader import CreateDataLoader
from models.optimize_model import OptimizeModel
from utils.evaluator import Evaluator


def main():
    opt = OptOptions().parse()
    # distributed learning initiate
    if opt.dist:
        init_dist()
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1
        rank = -1
    opt.process_rank = rank
    if rank <= 0:
        init_opt(opt)
    
    # set data loader
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_dataset()
    dataset_size = len(data_loader)
    assert(len(dataset.dataset.all_datasets) == 1)
    opt_dataset = dataset.dataset.all_datasets[0]

    # init model
    model = OptimizeModel(opt)

    # important, sample data each time
    torch.manual_seed(int(time.time()))
    numpy.random.seed(int(time.time()))
    random.seed(int(time.time()))

    # evaluator
    evaluator = Evaluator(opt, opt_dataset, model)
    evaluator.clear()

    # run optimize
    num_iter = len(dataset)
    time_stat = TimeStat(num_iter)
    for iter_id, data in enumerate(dataset):
        time_stat.opt_iter_start()
        # run optimize
        model.set_input(data)
        model.init_optimize()
        model.optimize(iter_id, num_iter)
        # obtain results
        pred_res = model.get_pred_result()
        data_idxs = data['index'].numpy()
        evaluator.update(data_idxs, pred_res)
        time_stat.opt_iter_end()
        if rank <= 0:
            time_stat.print_stat()
    if rank <= 0:
        time_stat.print_stat(opt_complete=True)

    # gather result in distributed mode
    if opt.dist:
        res_pkl_file = f'.estimator_{opt.opt_dataset}_{opt.strategy}_{rank}.pkl'
        ry_utils.save_pkl(res_pkl_file, evaluator)
        torch.distributed.barrier()
        # gather
        if rank == 0:
            evaluator.clear()
            for i in range(world_size):
                res_pkl_file =  f'.estimator_{opt.opt_dataset}_{opt.strategy}_{i}.pkl'
                tmp_evaluator = ry_utils.load_pkl(res_pkl_file)
                evaluator.gather_pred(tmp_evaluator.pred_results)
                os.remove(res_pkl_file) # remove tmp file

    if rank <= 0:
        evaluator.remove_redunc()
        test_res_dir = 'evaluate_results/optimize'
        ry_utils.build_dir(test_res_dir)
        res_pkl_file = osp.join(test_res_dir, f'{opt.opt_dataset}.pkl')
        ry_utils.save_pkl(res_pkl_file, evaluator)

        # print results
        metrics = ['mpjpe_3d', 'inter_mpjpe_3d', 'collision_ave', 'collision_max']
        for metric in metrics:
            metric_res = getattr(evaluator, metric)
            print(f"{metric} : {metric_res:.3f} (optimize)")
        print()


if __name__ == '__main__':
    main()