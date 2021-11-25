
import os, sys, shutil
import os.path as osp
import cv2
import pdb
import itertools
from collections import OrderedDict
import time
import numpy as np
import smplx
import torch
import torch.nn as nn
import torch.nn.functional as F
import ry_utils
import pdb
from torch.nn.parallel import DistributedDataParallel
from .base_model import BaseModel
from models.networks import InterHandEncoder, InterHandSubNetwork
from .transform_utils import batch_orthogonal_project
from .loss_utils import LossUtil
from utils import vis_util
from utils import render_color_utils as rcu


class MLPModel(BaseModel):
    @property
    def name(self):
        return 'InterHandModel'

    def __init__(self, opt):
        BaseModel.initialize(self, opt)

        # set params
        self.inputSize = opt.inputSize
        self.batch_size = opt.batchSize

        self.total_params_dim = opt.total_params_dim
        self.cam_params_dim = opt.cam_params_dim
        self.pose_params_dim = opt.pose_params_dim
        self.shape_params_dim = opt.shape_params_dim
        self.trans_params_dim = opt.trans_params_dim
        assert self.total_params_dim == \
               self.cam_params_dim+ self.trans_params_dim + \
                   self.pose_params_dim+self.shape_params_dim

        # separate
        self.right_shape_params_dim = self.shape_params_dim // 2
        self.left_shape_params_dim = self.shape_params_dim // 2
        self.right_pose_params_dim = self.pose_params_dim//2 - 3
        self.left_pose_params_dim = self.pose_params_dim//2 - 3
        self.right_orient_dim = 3
        self.left_orient_dim = 3
        self.hand_trans_dim = 3

        # initialization for input
        self.init_input()

        # load mean params, the mean params are from HMR
        self.load_mean_params()

        # load mano models
        self.load_mano_model()

        # loss utils
        self.loss_util = LossUtil(opt, self.mano_models)

        # mlp networks
        self.sub_network_list = list()
        self.set_default_loss_weights()


    def load_mean_params(self):
        # load from file
        mean_param_file = osp.join(self.opt.model_root, 
            self.opt.mean_param_file)
        mean_vals = ry_utils.load_pkl(mean_param_file)
        # mean_params: [cam, r_pose, l_pose, r_betas, l_betas, hand_trans]
        mean_params = np.zeros((1, self.total_params_dim))
        # set camera model first
        mean_params[0, 0] = 5.0
        # set pose
        mean_pose = mean_vals['mean_pose']
        mean_pose[:3] = 0.
        mean_pose = mean_pose[None, :] # (1, 48)
        mean_pose = np.repeat(mean_pose, 2, axis=0) # (2, 48)
        mean_pose = mean_pose.reshape(1, self.pose_params_dim) # (1, 96)
        # set shape
        mean_shape = mean_vals['mean_betas'].reshape(1, 10)
        mean_shape = np.repeat(mean_shape, 2, axis=0) # (2, 10)
        mean_shape = mean_shape.reshape(1, self.shape_params_dim) # (1, 20)
        # set trans
        mean_hand_trans = np.zeros((1, 3), dtype=np.float32)
        # concat
        mean_params[0, 3:] = np.hstack((mean_pose, mean_shape, mean_hand_trans))
        mean_params = np.repeat(mean_params, self.batch_size, axis=0)
        self.mean_params = torch.from_numpy(mean_params).float()
        self.mean_params.requires_grad = False

    
    def load_mano_model(self):
        # define finger tips index
        self.joint_ids = torch.tensor([744, 320, 443, 554, 671]).long().cuda()
        # load mano model 
        mano_models = dict()
        for hand_type in ['left', 'right']:
            mano_model_file = osp.join(self.opt.model_root, f"MANO_{hand_type.upper()}.pkl")
            is_rhand = (hand_type == 'right')
            mano_model = smplx.create( mano_model_file, 'mano', 
                use_pca=False, is_rhand=is_rhand, batch_size=self.batch_size * 2)
            mano_models[hand_type] = mano_model
        
        mano_shapedirs_left = mano_models['left'].shapedirs # (778, 3, 10)
        mano_shapedirs_right = mano_models['right'].shapedirs
        shape_diff = torch.mean(torch.abs(mano_shapedirs_left[:,0,:] -  mano_shapedirs_right[:,0,:]))
        if shape_diff < 1e-7:
            mano_models['left'].shapedirs[:, 0, :] *= -1
        # to cuda
        self.mano_models = dict()
        for key in mano_models:
            self.mano_models[key] = mano_models[key].cuda()


    def init_input(self):
        nb = self.batch_size
        opt = self.opt

        # set input image
        self.input_img = self.Tensor(
            nb, opt.input_nc, self.inputSize, self.inputSize)

        # hand class
        self.hand_type_array = self.Tensor(nb, 2)
        self.hand_type_valid = self.Tensor(nb, 1)
        # joints 2d & 3d
        self.joints_2d = self.Tensor(nb, opt.num_joints, 3)
        self.joints_3d = self.Tensor(nb, opt.num_joints, 4)
        # mano pose params
        self.gt_pose_params = self.Tensor(nb, opt.pose_params_dim)
        self.gt_shape_params = self.Tensor(nb, opt.shape_params_dim)
        self.mano_params_weight = self.Tensor(nb, 2)
        # set hand translation
        self.hand_trans = self.Tensor(nb, 1, 4)
        # data indexs
        self.data_idxs = self.Tensor(nb)

        # img features, init joints, and init joints (2d & 3d)
        self.img_feat = self.Tensor(nb, 1024)
        self.init_joints_2d = self.Tensor(nb, opt.num_joints, 3)
        self.init_joints_3d = self.Tensor(nb, opt.num_joints, 4)
        self.init_cam = self.Tensor(nb, 3)
        self.init_pose_params = self.Tensor(nb, opt.pose_params_dim)
        self.init_shape_params = self.Tensor(nb, opt.shape_params_dim)
        self.init_hand_trans = self.Tensor(nb, 3)
        self.init_right_pose_params = self.Tensor(nb, opt.pose_params_dim//2)
        self.init_left_pose_params = self.Tensor(nb, opt.pose_params_dim//2)
        self.init_right_shape_params = self.Tensor(nb, opt.shape_params_dim//2)
        self.init_left_shape_params = self.Tensor(nb, opt.shape_params_dim//2)


    def set_input(self, input):
         # image
        input_img = input['img']
        self.input_img.resize_(input_img.size()).copy_(input_img)
        # hand type
        hand_type_array = input['hand_type_array']
        hand_type_valid = input['hand_type_valid']
        self.hand_type_array.resize_(hand_type_array.size()).copy_(hand_type_array)
        self.hand_type_valid.resize_(hand_type_valid.size()).copy_(hand_type_valid)
        # gt joints 2d
        joints_2d = input['joints_2d']
        self.joints_2d.resize_(joints_2d.size()).copy_(joints_2d)
        # joints 3d
        joints_3d = input['joints_3d']
        self.joints_3d.resize_(joints_3d.size()).copy_(joints_3d)
        # hand trans
        hand_trans = input['hand_trans']
        self.hand_trans.resize_(hand_trans.size()).copy_(hand_trans)
        # mano pose
        mano_pose = input['mano_pose']
        mano_betas = input['mano_betas']
        mano_params_weight = input['mano_params_weight']
        self.gt_pose_params.resize_(mano_pose.size()).copy_(mano_pose)
        self.gt_shape_params.resize_(mano_betas.size()).copy_(mano_betas)
        self.mano_params_weight.resize_(
            mano_params_weight.size()).copy_(mano_params_weight)
        # index
        index = input['index']
        self.data_idxs.resize_(index.size()).copy_(index)
        self.data_idxs = self.data_idxs.long()

        # init predictions & image feature
        img_feat = input['img_feat']
        self.img_feat.resize_(img_feat.size()).copy_(img_feat)
        # init joints 2d & 3d
        init_joints_2d = input['init_joints_2d']
        self.init_joints_2d.resize_(init_joints_2d.size()).copy_(init_joints_2d)
        init_joints_3d = input['init_joints_3d']
        self.init_joints_3d.resize_(init_joints_3d.size()).copy_(init_joints_3d)
        # init mano params & camera params
        init_cam = input['init_cam']
        self.init_cam.resize_(init_cam.size()).copy_(init_cam)
        init_pose_params = input['init_pose_params']
        self.init_pose_params.resize_(init_pose_params.size()).copy_(init_pose_params)
        init_shape_params = input['init_shape_params']
        self.init_shape_params.resize_(init_shape_params.size()).copy_(init_shape_params)
        # hand trans (from motion prediction)
        init_hand_trans = input['init_hand_trans']
        self.init_hand_trans.resize_(init_hand_trans.size()).copy_(init_hand_trans)

        # split params into right / left hand
        self.init_right_orient = self.init_pose_params[:, :3]
        self.init_left_orient = self.init_pose_params[:, 48:48+3]
        self.init_right_pose_params = self.init_pose_params[:, 3:48]
        self.init_left_pose_params = self.init_pose_params[:, 48+3:]
        self.init_right_shape_params = self.init_shape_params[:, :10]
        self.init_left_shape_params = self.init_shape_params[:, 10:]

    
    def set_default_loss_weights(self):
        self.default_loss_weights = dict(
            joints_2d_loss = 10.0,
            joints_3d_loss = 10.0,
            mano_pose_loss = 10.0,
            mano_shape_loss = 10.0,
            hand_trans_loss = 10.0,
            shape_reg_loss = 0.1,
            shape_residual_loss = 1.0,
            collision_loss = 1.0
        )
        collision_weight = self.default_loss_weights['collision_loss']
        assert np.abs(collision_weight-1.0)<1e-7
        

    def get_mano_output(self, 
        right_orient, left_orient,
        right_pose_params, left_pose_params,
        right_shape_params, left_shape_params, hand_trans):

        output_verts = dict()
        output_joints = dict()

        left_orient_f = left_orient.clone()
        left_orient_f[:, 1] *= -1
        left_orient_f[:, 2] *= -1

        left_pose_params_f = left_pose_params.clone()
        left_pose_params_f = left_pose_params_f.reshape(self.batch_size*15, 3)
        left_pose_params_f[:, 1] *= -1
        left_pose_params_f[:, 2] *= -1
        left_pose_params_f = left_pose_params_f.reshape(self.batch_size, 15*3)

        hand_orient = torch.cat([right_orient, left_orient_f], dim=0)
        hand_pose_params = torch.cat([right_pose_params, left_pose_params_f], dim=0)
        hand_shape_params = torch.cat([right_shape_params, left_shape_params], dim=0)

        output = self.mano_models['right'](
            global_orient = hand_orient,
            hand_pose = hand_pose_params,
            betas = hand_shape_params,
        )
        vertices = output.vertices
        joints = output.joints
        extra_joints = torch.index_select(vertices, 1, self.joint_ids)
        joints = torch.cat([joints, extra_joints], dim=1)

        output_verts['right'] = vertices[:self.batch_size, ...]
        output_joints['right'] = joints[:self.batch_size, ...]

        # recover left verts
        left_verts = vertices[self.batch_size:, ]
        left_joints = joints[self.batch_size:, ]
        left_verts[:, :, 0] *= -1
        left_joints[:, :, 0] *= -1
        output_verts['left'] = left_verts
        output_joints['left'] = left_joints

        # apply trans and merge joints
        left_hand_verts = output_verts['left']
        right_hand_verts = output_verts['right']
        left_hand_joints = output_joints['left']
        right_hand_joints = output_joints['right']

        # move left hand to right wrist
        left_to_right_shift = right_hand_joints[:, 0:1, :] - left_hand_joints[:, 0:1, :] 
        # apply predicted translation
        hand_trans = hand_trans.view(self.batch_size, 1, 3) 
        shift = hand_trans + left_to_right_shift

        left_hand_verts = left_hand_verts + shift
        left_hand_joints = left_hand_joints + shift
        # merge
        joints = torch.cat([right_hand_joints, left_hand_joints], dim=1) # (bs, 42, 3)

        return right_hand_verts, left_hand_verts, joints

    
    def set_update_info(self, strategy, num_data):
        self.strategy = strategy

        self.update_loss_name_list = set()
        self.update_param_name_list = set()

        for stage in strategy:
            for param_name in stage['update_params']:
                self.update_param_name_list.add(param_name.replace('pred_', '')) # without pred_xx
            for loss_name, _ in stage['filter_loss']:
                self.update_loss_name_list.add(loss_name) 
            self.update_loss_name_list.add(stage['select_loss'])
        
        # pseudo forward to obtain shape of predicted params and losses
        self.forward(forward_backbone=True)
        self.compute_loss()

        # data idxs (existance)
        self.data_idxs_all = torch.zeros(num_data, dtype=torch.bool).cuda() # important !!!!, must be intialized again

        # features
        self.img_feat_all = torch.zeros((num_data, 1024), dtype=torch.float32).cuda()

        # params
        self.prev_params = dict()
        for param_name in self.update_param_name_list:
            pred_name = "pred_" + param_name
            prev_name = "prev_" + param_name
            param_dim = getattr(self, pred_name).size(1)
            self.prev_params[prev_name] = torch.zeros((num_data, param_dim), dtype=torch.float32).cuda()
            # print(prev_name, self.prev_params[prev_name].size())

        # losses
        self.prev_losses = dict()
        for loss_name in self.update_loss_name_list:
            pred_name = loss_name + "_batch"
            prev_name = "prev_" + loss_name + "_batch"
            self.prev_losses[prev_name] = torch.zeros(num_data, dtype=torch.float32).cuda()
        

    def save_pred_to_prev(self):
        # set data idxs
        self.data_idxs_all[self.data_idxs] = 1

        # set image features
        self.img_feat_all[self.data_idxs] = self.img_feat

        # update params
        for param_name in self.update_param_name_list:
            pred_name = "pred_" + param_name
            prev_name = "prev_" + param_name
            pred_param = getattr(self, pred_name).clone()
            self.prev_params[prev_name][self.data_idxs, :] = pred_param.clone()

        # update losses
        for loss_name in self.update_loss_name_list:
            pred_name = loss_name + "_batch"
            prev_name = "prev_" + loss_name + "_batch"
            pred_loss = getattr(self, pred_name).clone()
            self.prev_losses[prev_name][self.data_idxs] = pred_loss.clone()
    

    def __determine_input_dim(self):
        input_dim = self.opt.total_params_dim + 1024
        return input_dim

    def __get_param_dim(self, param_name):
        record = param_name.split('_')
        assert record[0] == 'pred'
        param_name = '_'.join(record[1:])
        param_dim = getattr(self, f"{param_name}_dim")
        return param_dim

    def add_new_network(self, stage_id):
        update_params = self.strategy[stage_id]['update_params']
        lr = self.strategy[stage_id]['lr']

        # obtain param dim
        param_dim_total = 0
        for param_name in update_params:
            param_dim = self.__get_param_dim(param_name)
            param_dim_total += param_dim

        # add new sub_networks
        input_dim = self.__determine_input_dim()
        sub_network = InterHandSubNetwork(self.opt, input_dim, param_dim_total).cuda()
        if self.opt.dist:
            sub_network = DistributedDataParallel(
                sub_network, device_ids=[torch.cuda.current_device()])
        self.sub_network_list.append(sub_network)

        # load pretrain weights
        if self.isTrain and self.opt.pretrain_weights_dir is not None:        
            pretrain_weights_dir = self.opt.pretrain_weights_dir
            pretrain_weights_file = osp.join(pretrain_weights_dir, f"pretrain_net_mlp_stage_{stage_id:02d}.pth")
            assert osp.exists(pretrain_weights_file), f"{pretrain_weights_file} does not exist."
            if self.opt.dist:
                sub_network.module.load_state_dict(
                    torch.load(pretrain_weights_file, map_location=lambda storage, 
                    loc: storage.cuda(torch.cuda.current_device())), strict=True)
            else:
                sub_network.load_state_dict(torch.load(pretrain_weights_file), strict=True)
            if self.opt.process_rank <= 0:
                print(f"Stage-{stage_id:02d} load pretrained weights.")

        # optimizer
        if self.isTrain:
            self.optimizer = torch.optim.Adam(
                sub_network.parameters(), lr=lr)
        

    def retrive_prev_prediction(self):
        # check existance of the data idxs
        assert torch.all(self.data_idxs_all[self.data_idxs])

        # update features
        self.img_feat = self.img_feat_all[self.data_idxs]

        # get params
        for param_name in self.update_param_name_list:
            pred_name = "pred_" + param_name
            prev_name = "prev_" + param_name
            cur_param = self.prev_params[prev_name][self.data_idxs, :].clone()
            setattr(self, pred_name, cur_param)

        # gather split params (pose / shape / etc into whole)
        self.__gather_params()


    def __gather_params(self):
        # concat shape params and pose params back
        shape_params_list = [self.pred_right_shape_params, self.pred_left_shape_params]
        self.pred_shape_params = torch.cat(shape_params_list, dim=1)
        # pose params
        pose_params_list = [
            self.pred_right_orient, self.pred_right_pose_params,
            self.pred_left_orient, self.pred_left_pose_params
        ]
        self.pred_pose_params = torch.cat(pose_params_list, dim=1)
        # final params
        self.final_params = torch.cat(
            [self.pred_cam_params, self.pred_pose_params, 
            self.pred_shape_params, self.pred_hand_trans], dim=1)
 

    def __forward_backbone(self):
        self.pred_pose_params = self.init_pose_params.clone()
        self.pred_shape_params = self.init_shape_params.clone()
        self.pred_cam_params = self.init_cam.clone()
        self.pred_hand_trans = self.init_hand_trans.clone()
        self.final_params = torch.cat([
            self.pred_cam_params, self.pred_pose_params, 
            self.pred_shape_params, self.pred_hand_trans ], dim=1)

        self.pred_right_orient = self.pred_pose_params[:, :3]
        self.pred_left_orient = self.pred_pose_params[:, 48:48+3]
        self.pred_right_pose_params = self.pred_pose_params[:, 3:48]
        self.pred_left_pose_params = self.pred_pose_params[:, 48+3:]
        self.pred_right_shape_params = self.pred_shape_params[:, :10]
        self.pred_left_shape_params = self.pred_shape_params[:, 10:]


    def __update_params_single(self, stage_id):
        if stage_id >= 0:
            # forward sub-network
            inputs = torch.cat([self.img_feat, self.final_params], dim=1)
            pred_residual = self.sub_network_list[stage_id](inputs)
            # update params (use pre-calculated replace information)
            param_dim_total = 0
            for param_name in self.strategy[stage_id]['update_params']:
                param_dim = self.__get_param_dim(param_name)
                old_params = getattr(self, f"{param_name}")
                residual = pred_residual[:, param_dim_total : param_dim_total+param_dim]
                new_params = old_params + residual
                param_dim_total += param_dim
                setattr(self, param_name, new_params)

    def __forward_mlp(self, stage_id):
        assert stage_id >= 0
        self.__update_params_single(stage_id)
        self.__gather_params() # __gather_params() should be call after each update_param_single


    def __forward_mano(self):
        # get pred verts and joints 3d
        self.pred_right_hand_verts, self.pred_left_hand_verts, self.pred_joints_3d = self.get_mano_output(
            self.pred_right_orient, self.pred_left_orient,
            self.pred_right_pose_params, self.pred_left_pose_params,
            self.pred_right_shape_params, self.pred_left_shape_params, self.pred_hand_trans)
        # generate predicted joints 2d
        self.pred_joints_2d = batch_orthogonal_project(
            self.pred_joints_3d, self.pred_cam_params)
        
        # split pose / betas into left / right && orient / finger
        self.gt_right_orient = self.gt_pose_params[:, :3]
        self.gt_left_orient = self.gt_pose_params[:, 48:48+3]
        self.gt_right_pose_params = self.gt_pose_params[:, 3:48]
        self.gt_left_pose_params = self.gt_pose_params[:, 48+3:]
        self.gt_right_shape_params = self.gt_shape_params[:, :10]
        self.gt_left_shape_params = self.gt_shape_params[:, 10:]
        # get gt verts and joints 3d
        self.gt_right_hand_verts, self.gt_left_hand_verts, self.gt_joints_3d_mano = self.get_mano_output(
            self.gt_right_orient, self.gt_left_orient,
            self.gt_right_pose_params, self.gt_left_pose_params,
            self.gt_right_shape_params, self.gt_left_shape_params, self.hand_trans[:, :, :3])


    def forward(self, forward_backbone=False, stage_id=-1):
        if forward_backbone:
            self.__forward_backbone()
        else:
            if stage_id < 0:
                stage_id = len(self.sub_network_list)-1
            self.__forward_mlp(stage_id)
        self.__forward_mano()


    def compute_loss(self, loss_weights=None):
        if loss_weights is None:
            loss_weights = self.default_loss_weights

        # joints 2d loss
        self.joints_2d_loss, _ = self.loss_util._joints_2d_loss(
            self.joints_2d[:, :, :2], self.pred_joints_2d, self.joints_2d[:, :, 2:3])
        self.joints_2d_loss = self.joints_2d_loss * loss_weights['joints_2d_loss']
        self.loss = self.joints_2d_loss

        # joints 2d loss (predicted 2d joints)
        _, self.joints_2d_loss_p_batch = self.loss_util._joints_2d_loss(
            self.init_joints_2d[:, :, :2], self.pred_joints_2d, self.init_joints_2d[:, :, 2:3])
        self.joints_2d_loss_p_batch *= loss_weights['joints_2d_loss']

        # joints_3d loss
        self.joints_3d_loss, _ = self.loss_util._joints_3d_loss(
            self.joints_3d[:, :, :3], self.pred_joints_3d, self.joints_3d[:, :, 3:4])
        self.joints_3d_loss = self.joints_3d_loss * loss_weights['joints_3d_loss']
        self.loss = (self.loss + self.joints_3d_loss)

        # joints_3d loss (predicted 3d joints)
        _, self.joints_3d_loss_p_batch = self.loss_util._joints_3d_loss(
            self.init_joints_3d[:, :, :3].clone(), self.pred_joints_3d, self.init_joints_3d[:, :, 3:4])
        self.joints_3d_loss_p_batch *= loss_weights['joints_3d_loss']

        # mano pose loss
        mano_right_pose_loss = self.loss_util._mano_pose_loss(self.gt_right_pose_params,
            self.pred_right_pose_params, self.mano_params_weight[:, 0:1])
        mano_left_pose_loss = self.loss_util._mano_pose_loss(self.gt_left_pose_params,
            self.pred_left_pose_params, self.mano_params_weight[:, 1:2])
        self.mano_pose_loss = mano_right_pose_loss + mano_left_pose_loss
        self.mano_pose_loss = self.mano_pose_loss * loss_weights['mano_pose_loss']
        self.loss = (self.loss + self.mano_pose_loss)

        # mano shape loss
        mano_right_shape_loss = self.loss_util._mano_shape_loss(self.gt_right_shape_params,
            self.pred_right_shape_params, self.mano_params_weight[:, 0:1])
        mano_left_shape_loss = self.loss_util._mano_shape_loss(self.gt_left_shape_params,
            self.pred_left_shape_params, self.mano_params_weight[:, 1:2])
        self.mano_shape_loss = mano_right_shape_loss + mano_left_shape_loss
        self.mano_shape_loss = self.mano_shape_loss * loss_weights['mano_shape_loss']
        self.loss = (self.loss + self.mano_shape_loss)

        # hand translation loss
        self.hand_trans_loss = self.loss_util._hand_trans_loss(
            self.hand_trans[:, 0, :3], self.pred_hand_trans, self.hand_trans[:, :, 3:4])
        self.hand_trans_loss = self.hand_trans_loss * loss_weights['hand_trans_loss']
        self.loss = (self.loss + self.hand_trans_loss)
    
        # shape constrain loss
        self.shape_reg_loss = self.loss_util._shape_reg_loss(self.pred_shape_params)
        self.shape_reg_loss = self.shape_reg_loss * loss_weights['shape_reg_loss']
        self.loss = (self.loss + self.shape_reg_loss)

        # shape deform loss
        right_shape_residual_loss = self.loss_util._shape_residual_loss(
            self.pred_right_shape_params, self.init_right_shape_params)
        left_shape_residual_loss = self.loss_util._shape_residual_loss(
            self.pred_left_shape_params, self.init_left_shape_params)
        self.shape_residual_loss = right_shape_residual_loss + left_shape_residual_loss
        self.shape_residual_loss = self.shape_residual_loss * loss_weights['shape_residual_loss']
        self.loss = (self.loss + self.shape_residual_loss)

        # collision loss
        self.collision_loss, self.collision_loss_batch, self.collision_loss_origin_scale = \
            self.loss_util._collision_loss(self.pred_right_hand_verts, self.pred_left_hand_verts, self.hand_type_array)
        self.collision_loss = self.collision_loss * loss_weights['collision_loss']
        self.collision_loss_batch = self.collision_loss_batch * loss_weights['collision_loss']
        self.loss = (self.loss + self.collision_loss)


    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


    def select_better_params(self, stage_id):
        # current stage
        stage = self.strategy[stage_id]

        # filter loss
        idxs = torch.ones(self.batch_size, dtype=torch.bool).cuda()
        for loss_name, percent in stage['filter_loss']:
            prev_name = f"prev_{loss_name}_batch"
            prev_loss = self.prev_losses[prev_name][self.data_idxs]
            cur_loss = getattr(self, f"{loss_name}_batch")
            idxs0 = cur_loss < prev_loss * (1+float(percent)/100)
            idxs = (idxs & idxs0)

        # select loss
        loss_name = stage['select_loss']
        prev_name = f"prev_{loss_name}_batch"
        prev_loss = self.prev_losses[prev_name][self.data_idxs]
        cur_loss = getattr(self, f"{loss_name}_batch")
        idxs0 = cur_loss <= prev_loss
        idxs = (idxs & idxs0)
        replace_idxs = ~idxs # replace current params / losses with previous losses

        # update params (TODO: use stage['update_params'] should be the same)
        # Since those params without updating are still the same.
        for pred_name in stage['update_params']:
            prev_name = pred_name.replace("pred_", "prev_")
            pred_param = getattr(self, pred_name)
            prev_param = self.prev_params[prev_name][self.data_idxs]
            pred_param[replace_idxs] = prev_param[replace_idxs]
            setattr(self, pred_name, pred_param)

        # update losses
        for loss_name in self.update_loss_name_list:
            pred_name = loss_name + "_batch"
            prev_name = "prev_" + loss_name + "_batch"
            pred_loss = getattr(self, pred_name)
            prev_loss = self.prev_losses[prev_name][self.data_idxs]
            pred_loss[replace_idxs] = prev_loss[replace_idxs]
            setattr(self, loss_name, pred_name)
            # print("!!!!!!!!!!!", loss_name, torch.mean(pred_loss))
        
        # after use prev params, reset data_idxs_all to zero
        self.data_idxs_all[self.data_idxs] = 0

        # gather params
        self.__gather_params()
    

    def __save_pred(self, save_dir):
        # params
        saved_params = dict()
        for param_name in self.prev_params:
            saved_params[param_name] = self.prev_params[param_name].detach().cpu()
        # losses
        saved_losses = dict()
        for loss_name in self.prev_losses:
            saved_losses[loss_name] = self.prev_losses[loss_name].detach().cpu()
        # idxs exist
        saved_data_idxs = self.data_idxs_all.detach().cpu()
        # dict
        res_dict = dict(
            saved_data_idxs = saved_data_idxs,
            saved_params = saved_params,
            saved_losses = saved_losses,
        )
        # save
        res_file = osp.join(save_dir, f"process_{self.opt.process_rank}.pkl")
        ry_utils.save_pkl(res_file, res_dict)
    
    def __gather_pred(self, save_dir):
        pkl_files = ry_utils.get_all_files(save_dir, ".pkl", "full")
        assert len(pkl_files) == torch.distributed.get_world_size()
        for pkl_file in pkl_files:
            rank_id = pkl_file.split('/')[-1].split('.')[0].split('_')[1]
            saved_info = ry_utils.load_pkl(pkl_file)
            data_idxs = saved_info['saved_data_idxs']
            saved_params = saved_info['saved_params']
            saved_losses = saved_info['saved_losses']
            self.data_idxs_all[data_idxs] = 1
            for param_name in saved_params:
                self.prev_params[param_name][data_idxs] = saved_params[param_name][data_idxs].cuda()
            for loss_name in saved_losses:
                self.prev_losses[loss_name][data_idxs] = saved_losses[loss_name][data_idxs].cuda()

    def sync(self, save_dir):
        if self.opt.dist:
            self.__save_pred(save_dir)
            torch.distributed.barrier()
            self.__gather_pred(save_dir)


    def test(self):
        with torch.no_grad():
            # forward backbone
            self.forward(forward_backbone=True)
            self.compute_loss()
            self.save_pred_to_prev()

            # forward MLPs
            for stage_id, stage in enumerate(self.strategy):
                self.retrive_prev_prediction() 
                self.forward(stage_id=stage_id)
                self.compute_loss() # calculate loss, without backward
                self.select_better_params(stage_id)
                self.save_pred_to_prev()
            
            self.__forward_mano() # forward mano after obtaining results
            self.compute_loss() # compute loss to obtain collision_loss_origin_scale


    def get_pred_result(self):
        pred_result = OrderedDict(
            pred_cam_params = self.pred_cam_params.cpu().numpy(),
            pred_pose_params = self.pred_pose_params.cpu().numpy(),
            pred_shape_params = self.pred_shape_params.cpu().numpy(),
            pred_hand_trans = self.pred_hand_trans.cpu().numpy(),
            gt_right_hand_verts = self.gt_right_hand_verts.cpu().numpy(),
            gt_left_hand_verts = self.gt_left_hand_verts.cpu().numpy(),
            pred_right_hand_verts = self.pred_right_hand_verts.cpu().numpy(),
            pred_left_hand_verts = self.pred_left_hand_verts.cpu().numpy(),
            mano_params_weight = self.mano_params_weight.cpu().numpy(),
            pred_joints_3d = self.pred_joints_3d.cpu().numpy(),
            gt_joints_3d = self.joints_3d.cpu().numpy(),
            do_flip = np.zeros(self.batch_size).astype(np.int32),
            collision_loss = self.collision_loss_batch.cpu().numpy(),
            collision_loss_origin_scale = self.collision_loss_origin_scale.cpu().numpy(),
        )
        return pred_result


    def get_current_errors(self):
        joints_2d_loss = self.joints_2d_loss.item()
        loss_dict = OrderedDict([('joints_2d_loss', joints_2d_loss)])

        joints_3d_loss = self.joints_3d_loss.item()
        loss_dict['joints_3d_loss'] = joints_3d_loss

        mano_pose_loss = self.mano_pose_loss.item()
        loss_dict['mano_pose_loss'] = mano_pose_loss

        mano_shape_loss = self.mano_shape_loss.item()
        loss_dict['mano_shape_loss'] = mano_shape_loss

        hand_trans_loss = self.hand_trans_loss.item()
        loss_dict['hand_trans_loss'] = hand_trans_loss

        shape_reg_loss = self.shape_reg_loss.item()
        loss_dict['shape_reg_loss'] = shape_reg_loss

        collision_loss = self.collision_loss.item()
        loss_dict['collision_loss'] = collision_loss
        
        shape_residual_loss = self.shape_residual_loss.item()
        loss_dict['shape_residual_loss'] = shape_residual_loss

        total_loss = self.loss.item()
        loss_dict['total_loss'] = total_loss

        loss_dict['hand_type_loss'] = 0.0

        return loss_dict


    def get_current_visuals(self, idx=0):
        assert self.opt.isTrain, "This function should not be called in test"

        # visualize image first
        img = self.input_img[idx].cpu().detach().numpy()
        show_img = vis_util.recover_img(img)[:,:,::-1]
        show_img_concat = np.concatenate((show_img, show_img), axis=1)
        size = self.opt.inputSize
        # show_img = cv2.resize(show_img, (size*2, size*2))
        visual_dict = OrderedDict([('img', show_img_concat)])

        # visualize keypoint
        kp = self.joints_2d[idx][:, :2].cpu().detach().numpy()

        pred_kp = self.pred_joints_2d[idx][:, :2].cpu().detach().numpy()
        kp_weight = self.joints_2d[idx][:, 2:].cpu().detach().numpy()

        kp_img = vis_util.draw_keypoints(
            img, kp, kp_weight, 'red', self.inputSize)
        pred_kp_img = vis_util.draw_keypoints(
            img, pred_kp, kp_weight, 'green', self.inputSize)
        kp_img = np.concatenate((kp_img, pred_kp_img), axis=1)

        # camera    
        cam = self.pred_cam_params[idx].cpu().detach().numpy()

        # visualize gt verts
        # left
        gt_left_vert = self.gt_left_hand_verts[idx].cpu().detach().numpy()
        gt_left_render_img = vis_util.render_mesh_to_image(
            self.opt.inputSize, img, cam, gt_left_vert, self.mano_models['left'].faces)
        gt_left_render_img = gt_left_render_img[:, :, ::-1]
        # right
        gt_right_vert = self.gt_right_hand_verts[idx].cpu().detach().numpy()
        gt_right_render_img = vis_util.render_mesh_to_image(
            self.opt.inputSize, img, cam, gt_right_vert, self.mano_models['right'].faces)
        gt_right_render_img = gt_right_render_img[:, :, ::-1]
        # concate
        gt_render_img = np.concatenate((gt_right_render_img, gt_left_render_img), axis=1)
        # render together
        verts_list = [gt_right_vert, gt_left_vert]
        faces_list = [self.mano_models['right'].faces, self.mano_models['left'].faces]
        color0 = np.array(rcu.colors['light_green']).reshape(1, 3)
        color1 = np.array(rcu.colors['light_blue']).reshape(1, 3)
        color_list = [color0, color1]
        gt_render_img_together = rcu.render_together(
            verts_list, faces_list, color_list, cam, self.opt.inputSize, show_img)

        # pred verts
        # left
        pred_left_vert = self.pred_left_hand_verts[idx].cpu().detach().numpy()
        pred_left_render_img = vis_util.render_mesh_to_image(
            self.opt.inputSize, img, cam, pred_left_vert, self.mano_models['left'].faces)
        pred_left_render_img = pred_left_render_img[:, :, ::-1]
        # right
        pred_right_vert = self.pred_right_hand_verts[idx].cpu().detach().numpy()
        pred_right_render_img = vis_util.render_mesh_to_image(
            self.opt.inputSize, img, cam, pred_right_vert, self.mano_models['right'].faces)
        pred_right_render_img = pred_right_render_img[:, :, ::-1]
        # concat
        pred_render_img = np.concatenate((pred_right_render_img, pred_left_render_img), axis=1)
        # render together
        verts_list = [pred_right_vert, pred_left_vert]
        faces_list = [self.mano_models['right'].faces, self.mano_models['left'].faces]
        color0 = np.array(rcu.colors['light_green']).reshape(1, 3)
        color1 = np.array(rcu.colors['light_blue']).reshape(1, 3)
        color_list = [color0, color1]
        pred_render_img_together = rcu.render_together(
            verts_list, faces_list, color_list, cam, self.opt.inputSize, show_img)
        
        render_img_together = np.concatenate([gt_render_img_together, pred_render_img_together], axis=1)[:, :, ::-1]

        visual_dict['gt_render_img (separate)'] = gt_render_img
        visual_dict['pred_render_img (separate)'] = pred_render_img
        visual_dict['render together (gt / pred)'] = render_img_together
        visual_dict['keypoint (gt / pred)'] = kp_img
        return visual_dict


    def save(self, epoch, stage_id):
        assert stage_id == len(self.sub_network_list)-1
        self.save_network(self.sub_network_list[stage_id], "mlp", epoch, stage_id)
        save_info = {'epoch': epoch,
                     'optimizer': self.optimizer.state_dict()}
        self.save_info(save_info, epoch, stage_id)
    
    
    def load(self, epoch, stage_id):
        assert stage_id == len(self.sub_network_list)-1
        load_success = self.load_network(
            self.sub_network_list[stage_id], "mlp", epoch, stage_id)
        return load_success


    def eval(self):
        for network in self.sub_network_list:
            network.eval()
    

    def update_learning_rate(self, epoch, stage_id):
        assert stage_id == len(self.sub_network_list)-1
        lr = self.strategy[stage_id]['lr']
        lr_decay_type = self.strategy[stage_id]['lr_decay_type']

        if lr_decay_type == 'cosine':
            old_lr = lr
            lr = 0.5*(1.0 + np.cos(np.pi*epoch/self.opt.total_epoch)) * old_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            assert lr_decay_type == 'none' # do nothing
            lr = lr

        if self.opt.process_rank <= 0:
            print("Current Learning Rate:{0:.2E}".format(lr))