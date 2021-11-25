import os
import os.path as osp
import sys
import shutil
import time
from datetime import datetime
import torch
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
from models.baseline_model import InterHandModel
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
    
    # set data loader
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_dataset()
    dataset_size = len(data_loader)

    # init model
    model = InterHandModel(opt)

    # set auxiliary class
    time_stat = TimeStat(opt.total_epoch)
    if rank <= 0:
        visualizer = Visualizer(opt)
        total_steps = 0
        print_count = 0
        loss_stat = LossStat(len(data_loader))
    
    if rank <= 0:
        vis_dir = osp.join(opt.checkpoints_dir, "visualization")
        ry_utils.renew_dir(vis_dir)

    # start training
    for epoch in range(opt.epoch_count, opt.total_epoch+1):

        epoch_iter = 0
        # important, sample data each time
        torch.manual_seed(int(time.time()))
        np.random.seed(int(time.time()))
        random.seed(int(time.time()))
        data_loader.shuffle_data()

        time_stat.epoch_init(epoch)
        if rank <= 0:
            loss_stat.set_epoch(epoch)

        for i, data in enumerate(dataset):
            model.set_input(data)
            time_stat.stat_data_time()
            model.forward()
            model.optimize_parameters()
            time_stat.stat_forward_time()

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
                    save_vis_result(vis_dict, vis_dir, epoch, i)
                    visualizer.plot_current_errors(epoch, float(
                        epoch_iter)/dataset_size, opt, errors)
            # print training time
            time_stat.stat_visualize_time()

        if rank <= 0:
            if epoch % opt.save_epoch_freq == 0:
                print( f"saving the model at the end of epoch {epoch}, iters {total_steps}")
                model.save(epoch, epoch)
            model.save("latest", epoch) # save latest weights at the end of each epoch
            time_stat.stat_epoch_time()
            time_stat.print_stat()

        # update learning rate
        model.update_learning_rate(epoch)

if __name__ == '__main__':
    main()