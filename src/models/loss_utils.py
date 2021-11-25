
import os, sys, shutil
import os.path as osp
import numpy as np
from collections import OrderedDict
import itertools
import torch.nn.functional as F
import torch.nn as nn
import torch
import pdb
import cv2
from .transform_utils import batch_rodrigues
from sdf import SDFLoss, SDFLoss_Single


class LossUtil(object):

    def __init__(self, opt, mano_models):
        self.inputSize = opt.inputSize
        self.pose_params_dim = opt.pose_params_dim // 2 # pose_params_dim for two hand
        self.isTrain = opt.isTrain
        if self.isTrain:
            self.use_hand_rotation = opt.use_hand_rotation
        else:
            self.use_hand_rotation = False
        self.batch_size = opt.batchSize
        """
        if self.isTrain:
            self.shape_reg_loss_format = opt.shape_reg_loss_format
        else:
            self.shape_reg_loss_format = 'l2'
        """

        faces_right = mano_models['right'].faces
        faces_left = mano_models['left'].faces
        robustifier = opt.sdf_robustifier if self.isTrain else None
        assert robustifier is None or robustifier > 0.0
        self.sdf_loss = SDFLoss(faces_right, faces_left, robustifier=robustifier).cuda()


    def _hand_type_loss(self, gt_hand_type, pred_hand_type, hand_type_valid):
        loss = F.binary_cross_entropy(pred_hand_type, gt_hand_type, reduction='none')
        loss = loss * hand_type_valid
        return torch.mean(loss)


    def _mano_pose_loss(self, mano_pose, pred_mano_pose, mano_params_weight):
        # change pose parameters to rodrigues matrix
        pose_dim = pred_mano_pose.size(1)
        assert pose_dim in [45, 48]
        
        pose_rodrigues = batch_rodrigues(mano_pose.contiguous().view(-1, 3)).view(
            self.batch_size, pose_dim//3, 3, 3)

        pred_pose_rodrigues = batch_rodrigues(pred_mano_pose.contiguous().view(-1, 3)).view(\
            self.batch_size, pose_dim//3, 3, 3)
        
        if not self.use_hand_rotation and pose_dim == 48: # pose-params contain global orient
            pose_params = pose_rodrigues[:, 1:, :, :].view(self.batch_size, -1)
            pred_pose_params = pred_pose_rodrigues[:, 1:, :, :].view(self.batch_size, -1)
        else:
            pose_params = pose_rodrigues.view(self.batch_size, -1)
            pred_pose_params = pred_pose_rodrigues.view(self.batch_size, -1)

        # square loss
        params_diff = pose_params - pred_pose_params
        square_loss = torch.mul(params_diff, params_diff)
        square_loss = square_loss * mano_params_weight
        loss = torch.mean(square_loss)

        return loss
    

    def _mano_shape_loss(self, mano_shape, pred_mano_shape, mano_params_weight):
        # abs loss
        shape_diff = torch.abs(mano_shape - pred_mano_shape)
        abs_loss = shape_diff * mano_params_weight
        loss = torch.mean(abs_loss)
        return loss
    

    def _joints_2d_loss(self, gt_keypoint, pred_keypoint, keypoint_weights):
        abs_loss = torch.abs((gt_keypoint-pred_keypoint))
        weighted_loss = abs_loss * keypoint_weights
        loss_batch = weighted_loss.reshape(self.batch_size, -1).mean(1)
        loss = torch.mean(weighted_loss)
        return loss, loss_batch


    def __align_by_root(self, joints_3d, joints_3d_weight):
        # right
        has_right_idxs = joints_3d_weight[:, 0, 0] > 0.5 # has right wrist
        joints_3d[has_right_idxs, :, :] = \
            joints_3d[has_right_idxs, :, :] - joints_3d[has_right_idxs, 0:1, :]
        # left
        no_right_idxs = joints_3d_weight[:, 0, 0] < 1e-7 # has no right wrist
        joints_3d[no_right_idxs, :, :] = \
            joints_3d[no_right_idxs, :, :] - joints_3d[no_right_idxs, 21:22, :]
        
    def _joints_3d_loss(self, gt_joints_3d, pred_joints_3d, joints_3d_weight):
        # align the root by default
        self.__align_by_root(gt_joints_3d, joints_3d_weight)
        self.__align_by_root(pred_joints_3d, joints_3d_weight)

        # calc squared loss
        joints_diff = gt_joints_3d - pred_joints_3d
        square_loss = torch.mul(joints_diff, joints_diff)
        square_loss = square_loss * joints_3d_weight
        loss_batch = square_loss.reshape(self.batch_size, -1).mean(1)
        loss = torch.mean(square_loss)
        return loss, loss_batch
    

    def _hand_trans_loss(self, gt_hand_trans, pred_hand_trans, hand_trans_weight):
        diff = gt_hand_trans - pred_hand_trans
        square_loss = diff * diff * hand_trans_weight
        loss = torch.mean(square_loss)
        return loss


    def _shape_reg_loss(self, shape_params):
        right_hand_shape = shape_params[:, :10]
        left_hand_shape = shape_params[:, 10:]
        diff = right_hand_shape - left_hand_shape
        losses = diff * diff # l2
        loss_batch = losses.reshape(self.batch_size, -1).mean(1)
        loss = torch.mean(losses)
        return loss
    

    def _shape_residual_loss(self, pred_shape_params, init_shape_params):
        diff = pred_shape_params - init_shape_params
        loss = torch.abs(diff)
        loss = torch.mean(loss)
        return loss
    

    def _finger_reg_loss(self, joints_3d):
        joint_idxs = [
            [1, 2, 3, 17], # index
            [4, 5, 6, 18,], # middle
            [7, 8, 9, 20,], # little
            [10, 11, 12, 19], # ring
            [13, 14, 15, 16], # thumb
        ]
        joint_idxs = np.concatenate(np.array(joint_idxs))
        joint_idxs = np.concatenate( [joint_idxs, (joint_idxs + 21)] )
        joint_idxs = torch.from_numpy(joint_idxs).long().cuda()

        bs = joints_3d.size(0)
        joints_3d = joints_3d[:, joint_idxs, :]
        joints_3d = joints_3d.view(bs, 10, 4, 3)
        joints_3d = joints_3d.view(bs*10, 4, 3)

        fingers = torch.zeros(bs*10, 3, 3).float()
        for i in range(3):
            fingers[:, i, :] = joints_3d[:, i, :] - joints_3d[:, i+1, :]

        cross_value1 = torch.cross(fingers[:, 0, :], fingers[:, 1, :], dim=1)
        C1 = torch.sum(fingers[:, 2, :] * cross_value1, dim=1)

        cross_value2 = torch.cross(fingers[:, 1, :], fingers[:, 2, :], dim=1)
        C2 = torch.sum(cross_value1 * cross_value2, dim=1)
        zero_pad = torch.zeros(C2.size()).float()
        loss = torch.abs(C1) - torch.min(zero_pad, C2)
        loss = loss.view(bs, 10)

        loss_batch = torch.sum(loss, dim=1)
        loss = torch.mean(loss_batch)

        return loss, loss_batch
    

    def _collision_loss(self, right_hand_verts, left_hand_verts, hand_type_array):
        bs = self.batch_size

        right_hand_verts = right_hand_verts.unsqueeze(dim=1) # (bs, 1, 778, 3)
        left_hand_verts = left_hand_verts.unsqueeze(dim=1) # (bs, 1, 778, 3)
        hand_verts = torch.cat([right_hand_verts, left_hand_verts], dim=1)

        losses, _, losses_origin_scale = self.sdf_loss(
            hand_verts, return_per_vert_loss=True, return_origin_scale_loss=True)
        losses = losses.reshape(bs, 1)

        # weights
        weights = torch.sum(hand_type_array, dim=1)>1.5
        weights = weights.type(torch.float32).view(bs, 1)
        losses = losses * weights
        # losses_origin_scale = losses_origin_scale * weights

        loss = torch.mean(losses)
        loss_batch = losses.view(-1,)
        return loss, loss_batch, losses_origin_scale