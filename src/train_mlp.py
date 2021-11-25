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
from utils.train_utils import save_vis_result, AverageMeter, TimeStat, LossStat
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from utils.visualizer import Visualizer
from models.mlp_model import MLPModel
from strategies import strategies
from utils.init_utils import init_dist, init_opt


def main():
    opt = TrainOptions().parse()

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
    
    mid_res_dir = osp.join(opt.checkpoints_dir, "mid_res", "train")
    vis_dir = osp.join(opt.checkpoints_dir, "visualization")
    if rank <= 0:
        visualizer = Visualizer(opt)
        ry_utils.renew_dir(vis_dir)
        ry_utils.renew_dir(mid_res_dir)
    torch.distributed.barrier()

    # set data loader
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_dataset()
    dataset_size = len(data_loader)

    # init model & strategy
    model = MLPModel(opt)
    strategy = strategies[opt.strategy]
    model.set_update_info(strategy, dataset_size)

    # forward backbone to obtain init results
    with torch.no_grad():
        for i, data in enumerate(dataset):
            model.set_input(data)
            model.forward(forward_backbone=True)
            model.compute_loss()
            model.save_pred_to_prev()
        model.sync(mid_res_dir)
    
    for stage_id, stage in enumerate(strategy):
        # setting params to be updated && new networks
        model.add_new_network(stage_id)

        # set auxiliary class
        total_epoch = stage['epoch']
        time_stat = TimeStat(total_epoch, stage_id)
        total_steps = 0
        print_count = 0
        loss_stat = LossStat(len(data_loader))
        
        # start epoch
        for epoch in range(opt.epoch_count, total_epoch+1):
            epoch_iter = 0
            torch.manual_seed(int(time.time()))
            numpy.random.seed(int(time.time()))
            random.seed(int(time.time()))
            data_loader.shuffle_data()

            time_stat.epoch_init(epoch)
            loss_stat.set_epoch(epoch)

            for i, data in enumerate(dataset):
                model.set_input(data)
                time_stat.stat_data_time()
                model.retrive_prev_prediction() # must be call at each forward
                # forward
                model.forward()
                model.compute_loss(stage['loss_weights'])
                model.optimize_parameters()
                time_stat.stat_forward_time()
            
                # stat and visualize
                if rank <= 0:
                    total_steps += opt.batchSize * world_size
                    epoch_iter += opt.batchSize * world_size
                    # get training losses
                    errors = model.get_current_errors()
                    loss_stat.update(errors)
                    # print loss
                    if total_steps/opt.print_freq > print_count:
                        loss_stat.print_loss(epoch_iter)
                        print_count += 1
                        sys.stdout.flush()
                    # get visualization
                    if total_steps % opt.display_freq == 0:
                        vis_dict = model.get_current_visuals()
                        visualizer.display_current_results(
                            vis_dict, epoch)
                        save_vis_result(vis_dict, vis_dir, epoch, i, stage_id)
                        visualizer.plot_current_errors(epoch, float(
                            epoch_iter)/dataset_size, opt, errors)
                time_stat.stat_visualize_time()

            if rank <= 0:
                model.save("latest", stage_id) # save latest weights at the end of each epoch
                time_stat.stat_epoch_time()
                time_stat.print_stat()
                print(f"stage-{stage_id+1:02d}, epoch:{epoch:02d} completes")

            # update learning rate
            model.update_learning_rate(epoch, stage_id)

        # update selective information at the end of each stage
        with torch.no_grad():
            for i, data in enumerate(dataset):
                model.set_input(data)
                model.retrive_prev_prediction() # must be call at each forward
                model.forward()
                model.compute_loss()
                model.select_better_params(stage_id)
                model.save_pred_to_prev()
            model.sync(mid_res_dir)
        

if __name__ == '__main__':
    main()