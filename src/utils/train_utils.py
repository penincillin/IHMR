import os
import os.path as osp
import sys
import shutil
import time
from datetime import datetime
import subprocess as sp
import cv2
import numpy as np


def save_vis_result(vis_dict, vis_dir, epoch, i, stage_id=None):
    res_img = None
    for key, value in vis_dict.items():
        if res_img is None:
            res_img = value
        else:
            res_img = np.concatenate( (res_img, value), axis=0)
    if stage_id is None:
        res_path = osp.join(vis_dir, f"epoch_{epoch:03d}_iter_{i:04d}.png")
    else:
        res_path = osp.join(vis_dir, f"stage_{stage_id:02d}_epoch_{epoch:03d}_iter_{i:04d}.png")
    cv2.imwrite(res_path, res_img[:,:,::-1])


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LossStat(object):
    def __init__(self, num_data):
        self.total_losses = AverageMeter()
        self.hand_type_losses = AverageMeter()
        self.joints_2d_losses = AverageMeter()
        self.joints_3d_losses = AverageMeter()
        self.hand_trans_losses = AverageMeter()
        self.mano_pose_losses = AverageMeter()
        self.mano_shape_losses = AverageMeter()
        self.collision_losses = AverageMeter()
        self.shape_reg_losses = AverageMeter()
        self.num_data = num_data
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def update(self, errors):
        self.total_losses.update(errors['total_loss'])
        self.hand_type_losses.update(errors['hand_type_loss'])
        self.joints_2d_losses.update(errors['joints_2d_loss'])
        self.joints_3d_losses.update(errors['joints_3d_loss'])
        self.mano_pose_losses.update(errors['mano_pose_loss'])
        self.mano_shape_losses.update(errors['mano_shape_loss'])
        self.hand_trans_losses.update(errors['hand_trans_loss'])
        self.collision_losses.update(errors['collision_loss'])
        self.shape_reg_losses.update(errors['shape_reg_loss'])
    
    def print_loss(self, epoch_iter):
        print_content = 'Epoch:[{}][{}/{}]\t' + \
                                    'Total Loss {tot_loss.val:.4f}({tot_loss.avg:.4f})\t' + \
                                    'Joints 2D Loss {kp_loss.val:.4f}({kp_loss.avg:.4f})\t' + \
                                    'Joints 3D Loss {j3d_loss.val:.4f}({j3d_loss.avg:.4f})'
        print_content = print_content.format(self.epoch, epoch_iter, self.num_data,
            tot_loss=self.total_losses, kp_loss=self.joints_2d_losses, j3d_loss=self.joints_3d_losses)

        print_content += '\nEpoch:[{}][{}/{}]\t' + \
                        'MANO Pose Params Loss {mp_loss.val:.4f}({mp_loss.avg:.4f})\t' + \
                        'MANO Shape Params Loss {ms_loss.val:.4f}({ms_loss.avg:.4f})'
        print_content = print_content.format(self.epoch, epoch_iter, self.num_data,
            mp_loss=self.mano_pose_losses, ms_loss=self.mano_shape_losses)

        print_content += '\nEpoch:[{}][{}/{}]\t' + \
            'Hand Trans Loss {htr_loss.val:.4f}({htr_loss.avg:.4f})\t' + \
            'Shape Reg Loss {sr_loss.val:.4f}({sr_loss.avg:.4f})\t' + \
            'Collision Loss {col_loss.val:.4f}({col_loss.avg:.4f})'
        print_content = print_content.format(self.epoch, epoch_iter, self.num_data,
            htr_loss=self.hand_trans_losses, sr_loss = self.shape_reg_losses, col_loss=self.collision_losses)

        print(print_content)
        print('=======================================================================')


class TimeStat(object):
    def __init__(self, total_epoch=100, stage_id=None):
        self.data_time = AverageMeter()
        self.forward_time = AverageMeter()
        self.visualize_time = AverageMeter()
        self.total_time = AverageMeter()
        self.total_epoch = total_epoch
        self.stage_id = stage_id
    
    def epoch_init(self, epoch):
        self.data_time_epoch = 0.0
        self.forward_time_epoch = 0.0
        self.visualize_time_epoch = 0.0
        self.start_time = time.time()
        self.epoch_start_time = time.time()
        self.forward_start_time = -1
        self.visualize_start_time = -1
        self.epoch = epoch
    
    def stat_data_time(self):
        self.forward_start_time = time.time()
        self.data_time_epoch += (self.forward_start_time - self.start_time)

    def stat_forward_time(self):
        self.visualize_start_time = time.time()
        self.forward_time_epoch += (self.visualize_start_time - self.forward_start_time)
    
    def stat_visualize_time(self):
        visualize_end_time = time.time()
        self.start_time = visualize_end_time
        self.visualize_time_epoch += visualize_end_time - self.visualize_start_time
    
    def stat_epoch_time(self):
        epoch_end_time = time.time()
        self.epoch_time = epoch_end_time - self.epoch_start_time
    
    def print_stat(self):
        self.data_time.update(self.data_time_epoch)
        self.forward_time.update(self.forward_time_epoch)
        self.visualize_time.update(self.visualize_time_epoch)

        time_content = f"End of epoch {self.epoch} / {self.total_epoch} \t" \
                        f"Time Taken: data {self.data_time.avg:.2f}, " \
                        f"forward {self.forward_time.avg:.2f}, " \
                        f"visualize {self.visualize_time.avg:.2f}, " \
                        f"Total {self.epoch_time:.2f} \n" 
        if self.stage_id is None:
            time_content += f"Epoch {self.epoch} compeletes in {datetime.now().strftime('%Y-%m-%d:%H:%M:%S')}"
        else:
             time_content += (f"Stage-{self.stage_id:02d} && Epoch-{self.epoch} compeletes in " +\
                  f"{datetime.now().strftime('%Y-%m-%d:%H:%M:%S')}")
        print(time_content)