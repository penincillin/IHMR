import os, sys, shutil
import os.path as osp
import time
from datetime import datetime
import torch
import numpy
import random
import numpy as np
import pdb
import cv2
import ry_utils
from torch.multiprocessing import Process
import torch.distributed as dist
import torch.multiprocessing as mp

from utils import eval_utils
from strategies import strategies
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.mlp_model import MLPModel
from utils.evaluator import Evaluator
from utils.eval_utils import ResultStat
from utils.init_utils import init_dist

def main():
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
  
    # distributed learning initiate
    if opt.dist:
        init_dist()
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1
        rank = -1
    opt.process_rank = rank

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_dataset()
    dataset_size = len(data_loader)
    assert(len(dataset.dataset.all_datasets) == 1)
    test_dataset = dataset.dataset.all_datasets[0]

    model = MLPModel(opt)
    strategy = strategies[opt.strategy]
    model.set_update_info(strategy, dataset_size)

    # load each sub-network
    for stage_id, stage in enumerate(strategy):
        # setting params to be updated && new networks
        model.add_new_network(stage_id)
        load_success = model.load(opt.test_epoch, stage_id)
        assert load_success, f"Sub-Network-{stage_id} Load failed."
    model.eval()

    # evaluator
    evaluator = Evaluator(opt, test_dataset, model)
    evaluator.clear()

    for i, data in enumerate(dataset):
        if rank <= 0:
            print(f"Evaluating IHMR-OPT: {i} / {len(dataset)}")
        model.set_input(data)
        model.test()
        pred_res = model.get_pred_result()
        data_idxs = data['index'].numpy()
        evaluator.update(data_idxs, pred_res)
        sys.stdout.flush()

    if opt.dist:
        res_pkl_file = f'.mlp_{opt.test_dataset}_{rank}.pkl'
        ry_utils.save_pkl(res_pkl_file, evaluator)
        torch.distributed.barrier()
        # gather
        if rank == 0:
            evaluator.clear()
            for i in range(world_size):
                res_pkl_file =  f'.mlp_{opt.test_dataset}_{i}.pkl'
                tmp_evaluator = ry_utils.load_pkl(res_pkl_file)
                evaluator.gather_pred(tmp_evaluator.pred_results)
                os.remove(res_pkl_file) # remove tmp file

    if rank <= 0:
        # save to file
        evaluator.remove_redunc()
        test_res_dir = osp.join('evaluate_results', 'mlp')
        ry_utils.build_dir(test_res_dir)
        res_pkl_file = osp.join(test_res_dir, f'{opt.test_dataset}.pkl')
        ry_utils.save_pkl(res_pkl_file, evaluator)

        # print results (overall)
        Stater = ResultStat()
        all_metrics = [info[0] for info in Stater.result_info]
        for metric in all_metrics:
            Stater.update(metric, opt.test_epoch, getattr(evaluator, metric))
        Stater.print_best_results()
        sys.stdout.flush()


if __name__ == '__main__':
    main()
