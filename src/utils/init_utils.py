import os, sys, shutil
import os.path as osp
import torch
from torch.multiprocessing import Process
import torch.distributed as dist
import torch.multiprocessing as mp
import ry_utils


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def init_opt(opt):
    expr_dir = os.path.join(opt.checkpoints_dir)
    ry_utils.build_dir(opt.checkpoints_dir)