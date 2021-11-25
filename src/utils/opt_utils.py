import os, sys, shutil
import os.path as osp
import time
from datetime import datetime
import torch
import numpy
import random
import cv2
import numpy as np
import pdb
import ry_utils


class TimeStat(object):
    def __init__(self, iter_total):
        self.iter_total = iter_total
        self.iter_count = 0
        self.total_time = 0.0
    
    def opt_iter_start(self):
        self.opt_start_time = time.time()
    
    def opt_iter_end(self):
        self.opt_end_time = time.time()
        self.opt_iter_time = self.opt_end_time - self.opt_start_time
        self.total_time += self.opt_iter_time
        self.iter_count += 1

    def print_stat(self, opt_complete=False):
        speed = self.iter_count / self.total_time
        remain_time = (self.iter_total - self.iter_count) / speed
        ave_time_cost = self.total_time / self.iter_count

        if not opt_complete:
            print(f"Opt completes: {self.iter_count}/{self.iter_total}," 
                    f"iter time: {self.opt_iter_time:.2f} sec, remain requires: {remain_time/60:.2f} mins")
            print('-------------------------------------------')
        if opt_complete:
            cur_time = datetime.now().strftime('%Y-%m-%d:%H:%M:%S')
            tt_min = self.total_time/60
            tt_hour = self.total_time/3600
            print(f"Opt compeletes in {cur_time}, total time cost: {tt_min:.3f} mins ({tt_hour:.3f} hours)")


def save_pred_obj(res_dir, pred_result, mano_models, iter_id, data_id, opt_iter):
    right_verts = pred_result['pred_right_hand_verts'][0]
    left_verts = pred_result['pred_left_hand_verts'][0]
    right_faces = mano_models['right'].faces
    left_faces = mano_models['left'].faces
    verts = np.concatenate([right_verts, left_verts], axis=0)
    faces = np.concatenate([right_faces, left_faces+right_verts.shape[0]], axis=0)
    res_path = osp.join(res_dir, 
        f"iter_{iter_id:04d}_stage_{data_id:02d}_opt_iter_{opt_iter:04d}.obj")
    ry_utils.save_mesh_to_obj(res_path, verts, faces)


def check_valid_loss(loss_name):
    """
    This code is used to check whether the given loss_name is valid to be used in ..
    filtering and selecting
    """
    invalid_name_list = [
        'joints_3d_loss',
        'joints_2d_loss',
        'hand_trans_loss',
    ]
    return loss_name not in invalid_name_list


def gather_params_losses(mid_results, stage):
    num_res = len(mid_results)
    update_params = stage['update_params']
    cand_losses = [item[0] for item in stage['filter_loss']]
    cand_losses.append(stage['select_loss'])

    # assign space to all params first
    all_params = dict()
    for param_name in update_params:
        sample_param = mid_results[0][param_name]
        param_size = list(sample_param.size())
        new_size = [num_res,] + param_size
        all_params[param_name] = torch.zeros(new_size, dtype=torch.float32).cuda()
    # assign param to new params
    for i in range(num_res):
        for param_name in update_params:
            all_params[param_name][i] = mid_results[i][param_name]
    
    # assign space to all losses first
    all_losses = dict()
    for loss_name in cand_losses:
        sample_loss = mid_results[0][loss_name]
        new_size = [num_res,] + list(sample_loss.size())
        all_losses[loss_name] = torch.zeros(new_size, dtype=torch.float32).cuda()
    for i in range(num_res):
        for loss_name in cand_losses:
            all_losses[loss_name][i] = mid_results[i][loss_name]
    
    return all_params, all_losses


"""
Following code for filtering losses by different criterion
"""
def __get_idxs_one_filter(all_losses, origin_losses, loss_name, criterion):
    assert criterion[0] in ['+', '-']
    # assert float(criterion)>-100 and float(criterion)<100

    losses_to_filter = all_losses[loss_name]
    origin_loss = origin_losses[loss_name]

    percent = (float(criterion) + 0.1) / 100 # + 0.1 is for smooth
    bar = origin_loss * (1 + percent)
    idxs = losses_to_filter <= bar
    return idxs


def filter_by_losses(all_losses, filter_losses):
    assert len(filter_losses)>0

    # origin losses means the losses at the begining of optimization starts (stage wise)
    origin_losses = dict()
    for loss_name in all_losses:
        origin_losses[loss_name] = all_losses[loss_name][0].clone().reshape(1, -1) # (bs)

    # filter valid mid_results by each losses
    loss_names = list(all_losses.keys())
    sample_loss = all_losses[loss_names[0]] # get size of idxs (num_result, bs)
    idxs = torch.ones(sample_loss.size(), dtype=torch.bool).cuda()
    for loss_name, criterion in filter_losses:
        idxs0 = __get_idxs_one_filter(all_losses, origin_losses, loss_name, criterion)
        idxs = (idxs & idxs0)
    valid_idxs = idxs
    invalid_idxs = ~idxs

    # set value of invalid losses to zero
    inf_num = 100000000000.0
    for loss_name in all_losses:
        losses = all_losses[loss_name]
        losses[invalid_idxs] = inf_num
        losses[0] = origin_losses[loss_name] # keep the origin loss unchanged
    return all_losses


def select_params(all_params, all_losses, select_loss_name):
    select_loss = all_losses[select_loss_name]
    idxs = torch.argmin(select_loss, dim=0)

    select_params = dict()
    for param_name in all_params:
        params = all_params[param_name]
        bs = params.size(1)
        select_params[param_name] = params[idxs, torch.arange(bs).long(), ...]
    return select_params