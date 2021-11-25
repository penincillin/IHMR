import os, sys, shutil
import os.path as osp
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_skew(vec, batchSize):
    with torch.cuda.device(vec.get_device()):
        vec = vec.view(batchSize,3)
        res = torch.zeros(batchSize, 9).cuda()
        res[:,1] = -vec[:, 2]
        res[:,2] = vec[:, 1]
        res[:,3] = vec[:, 2]
        res[:,5] = -vec[:, 0]
        res[:,6] = -vec[:, 1]
        res[:,7] = vec[:, 0]
        return res.view(batchSize, 3, 3)


def batch_rodrigues(pose_params):
    with torch.cuda.device(pose_params.get_device()):
        # pose_params shape is (bs*24, 3)
        # angle shape is (batchSize*24, 1)
        angle = torch.norm(pose_params+1e-8, p=2, dim=1).view(-1, 1)
        # r shape is (batchSize*24, 3, 1)
        r = torch.div(pose_params, angle).view(angle.size(0), -1, 1)
        # r_T shape is (batchSize*24, 1, 3)
        r_T = r.permute(0,2,1)
        # cos and sin is (batchSize*24, 1, 1)
        cos = torch.cos(angle).view(angle.size(0), 1, 1)
        sin = torch.sin(angle).view(angle.size(0), 1, 1)
        # outer is (bs*24, 3, 3)
        outer = torch.matmul(r, r_T)
        eye = torch.eye(3).view(1,3,3)
        # eyes is (bs*24, 3, 3)
        eyes = eye.repeat(angle.size(0), 1, 1).cuda()
        # r_sk is (bs*24, 3, 3)
        r_sk = batch_skew(r, r.size(0))
        R = cos * eyes + (1 - cos) * outer + sin * r_sk
        # R shape is (bs*24, 3, 3)
        return R


def batch_orthogonal_project(X, camera):
    # camera is (batchSize, 1, 3)
    camera = camera.view(-1, 1, 3)
    # x_trans is (batchSize, 19, 2)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    # first res is (batchSize, 19*2)
    res = camera[:, :, 0] * X_trans.view(X_trans.size(0), -1)
    return res.view(X_trans.size(0), X_trans.size(1), -1)