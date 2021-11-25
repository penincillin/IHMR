import os, sys, shutil
import os.path as osp
import time
from datetime import datetime
import torch
import random
import cv2
import numpy as np
import pdb
from utils import eval_utils
from torch.multiprocessing import Process
import torch.distributed as dist
import torch.multiprocessing as mp
import ry_utils
from utils.init_utils import init_dist, init_opt
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.baseline_model import InterHandModel
from utils.evaluator import Evaluator
from utils.eval_utils import ResultStat


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

    # data loader
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_dataset()
    assert(len(dataset.dataset.all_datasets) == 1)
    test_dataset = dataset.dataset.all_datasets[0]

    # load trained models
    epoch = opt.test_epoch
    model = InterHandModel(opt)
    load_success = model.load_network(
        model.encoder, 'baseline', epoch)
    assert load_success, f"The weights to be evaluated (epoch:{epoch} does not exist"
    model.eval()

    # evaluator
    evaluator = Evaluator(opt, test_dataset, model)
    evaluator.clear()

    # inference
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
        pred_res = model.get_pred_result()
        data_idxs = data['index'].numpy()
        evaluator.update(data_idxs, pred_res)
        if rank <= 0:
            print(f"Evaluating IHMR-Baseline: {i} / {len(dataset)}")
        sys.stdout.flush()

    # gather result in distributed mode
    if opt.dist:
        res_pkl_file = f'.baseline_{opt.test_dataset}_{rank}.pkl'
        ry_utils.save_pkl(res_pkl_file, evaluator)
        torch.distributed.barrier()
        # gather
        if rank == 0:
            evaluator.clear()
            for i in range(world_size):
                res_pkl_file = f'.baseline_{opt.test_dataset}_{i}.pkl'
                tmp_evaluator = ry_utils.load_pkl(res_pkl_file)
                evaluator.gather_pred(tmp_evaluator.pred_results)
                os.remove(res_pkl_file) # remove tmp file

    if rank <= 0:
        # save to file
        evaluator.remove_redunc()
        test_res_dir = osp.join('evaluate_results', 'baseline')
        ry_utils.build_dir(test_res_dir)
        res_pkl_file = osp.join(test_res_dir, f'{opt.test_dataset}.pkl')
        ry_utils.save_pkl(res_pkl_file, evaluator)
    
        # print results
        Stater = ResultStat()
        all_metrics = [info[0] for info in Stater.result_info]
        for metric in all_metrics:
            Stater.update(metric, epoch, getattr(evaluator, metric))
        Stater.print_best_results()
        sys.stdout.flush()

if __name__ == '__main__':
    main()
