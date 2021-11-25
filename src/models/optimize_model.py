
import os, sys, shutil
import os.path as osp
import cv2
import pdb
import itertools
from collections import OrderedDict
import numpy as np
import smplx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from .base_model import BaseModel
from .transform_utils import batch_orthogonal_project
from .loss_utils import LossUtil
import time
from utils import vis_util
from utils import render_color_utils as rcu
import ry_utils
from strategies import strategies
import utils.opt_utils as opt_utils


class OptimizeModel(BaseModel):
    @property
    def name(self):
        return 'OptimizeModel'

    def __init__(self, opt):
        BaseModel.initialize(self, opt)

        self.process_rank = opt.process_rank

        # set params
        self.inputSize = opt.inputSize
        self.total_params_dim = opt.total_params_dim
        self.cam_params_dim = opt.cam_params_dim
        self.pose_params_dim = opt.pose_params_dim
        self.shape_params_dim = opt.shape_params_dim
        self.trans_params_dim = opt.trans_params_dim

        assert self.total_params_dim == \
               self.cam_params_dim+ self.trans_params_dim + \
                   self.pose_params_dim+self.shape_params_dim

        self.batch_size = opt.batchSize
        nb = self.batch_size

        # hand class
        self.hand_type_array = self.Tensor(nb, 2)
        self.hand_type_valid = self.Tensor(nb, 1)
        # joints 2d 
        self.joints_2d = self.Tensor(nb, opt.num_joints, 3)
        # joints 3d
        self.joints_3d = self.Tensor(nb, opt.num_joints, 4)
        # mano pose params
        self.gt_pose_params = self.Tensor(nb, opt.pose_params_dim)
        self.gt_shape_params = self.Tensor(nb, opt.shape_params_dim)
        self.mano_params_weight = self.Tensor(nb, 2)
        # set hand translation
        self.hand_trans = self.Tensor(nb, 1, 4)

        # init predictions
        self.init_cam = self.Tensor(nb, 3)
        self.init_pose_params = self.Tensor(nb, opt.pose_params_dim)
        self.init_shape_params = self.Tensor(nb, opt.shape_params_dim)
        self.init_hand_trans = self.Tensor(nb, 1, 3)
        self.init_joints_2d = self.Tensor(nb, opt.num_joints, 3)
        self.init_joints_3d = self.Tensor(nb, opt.num_joints, 4)
        self.init_hand_trans_j = self.Tensor(nb, 1, 3)

        # load mano models
        self.load_mano_model()

        # loss utils
        self.loss_util = LossUtil(opt, self.mano_models)

        # strategies
        self.strategy = strategies[opt.strategy]
        self.set_default_loss_weights()    


    def set_default_loss_weights(self):
        self.default_loss_weights = dict(
            joints_2d_loss = 10.0,
            joints_3d_loss = 1000.0,
            trans_loss_weight = 100.0,
            shape_reg_loss_weight = 0.1,
            collision_loss_weight = 1.0,
            finger_reg_loss_weight = 100000.0,
        )
        collision_weight = self.default_loss_weights['collision_loss_weight']
        assert np.abs(collision_weight-1.0)<1e-7


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


    def set_input(self, input):
        # hand type
        hand_type_array = input['hand_type_array']
        hand_type_valid = input['hand_type_valid']
        self.hand_type_array.resize_(hand_type_array.size()).copy_(hand_type_array)
        self.hand_type_valid.resize_(hand_type_valid.size()).copy_(hand_type_valid)
        # joints 2d
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

        # init predictions
        # cam
        init_cam = input['init_cam']
        self.init_cam.resize_(init_cam.size()).copy_(init_cam)
        # pose param
        init_pose_params = input['init_pose_params']
        self.init_pose_params.resize_(init_pose_params.size()).copy_(init_pose_params)
        # shape param
        init_shape_params = input['init_shape_params']
        self.init_shape_params.resize_(init_shape_params.size()).copy_(init_shape_params)
        # hand trans (from motion prediction)
        init_hand_trans = input['init_hand_trans']
        self.init_hand_trans.resize_(init_hand_trans.size()).copy_(init_hand_trans)
        # 2D joints (from joints prediction)
        init_joints_2d = input['init_joints_2d']
        self.init_joints_2d.resize_(init_joints_2d.size()).copy_(init_joints_2d)
        self.init_joints_2d.requires_grad = False
        # 3D joints (from joints prediction)
        init_joints_3d = input['init_joints_3d']
        self.init_joints_3d.resize_(init_joints_3d.size()).copy_(init_joints_3d)
        self.init_joints_3d.requires_grad = False
        # hand trans (from joints prediction)
        init_hand_trans_j = input['init_hand_trans_j']
        self.init_hand_trans_j.resize_(init_hand_trans_j.size()).copy_(init_hand_trans_j)
        self.init_hand_trans_j.requires_grad = False
        

    def get_mano_output(self, 
        right_orient, left_orient,
        right_pose_params, left_pose_params,
        right_shape_params, left_shape_params,
        hand_trans, hand_type_array):

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


    def init_optimize(self):
        # cam params
        self.pred_cam_params = self.init_cam.clone()
        # hand translation
        pred_hand_trans = self.init_hand_trans.clone()
        self.pred_hand_trans = pred_hand_trans[..., :3]
        # pose and shape params
        pred_pose_params = self.init_pose_params.clone()
        pred_shape_params = self.init_shape_params.clone()

        # split pose / betas into left / right && orient / finger
        self.pred_right_orient = pred_pose_params[:, :3]
        self.pred_left_orient = pred_pose_params[:, 48:48+3]
        self.pred_right_pose_params = pred_pose_params[:, 3:48]
        self.pred_left_pose_params = pred_pose_params[:, 48+3:]
        self.pred_right_shape_params = pred_shape_params[:, :10]
        self.pred_left_shape_params = pred_shape_params[:, 10:]


    def forward(self):
        # get pred verts and joints 3d
        self.pred_right_hand_verts, self.pred_left_hand_verts, self.pred_joints_3d = self.get_mano_output(
            self.pred_right_orient, self.pred_left_orient,
            self.pred_right_pose_params, self.pred_left_pose_params,
            self.pred_right_shape_params, self.pred_left_shape_params,
            self.pred_hand_trans, self.hand_type_array)

        # generate predicted joints 2d
        self.pred_joints_2d = batch_orthogonal_project(
            self.pred_joints_3d, self.pred_cam_params)
        
        # concat shape params and pose params back
        shape_params_list = [self.pred_right_shape_params, self.pred_left_shape_params]
        self.pred_shape_params = torch.cat(shape_params_list, dim=1)
        pose_params_list = [
            self.pred_right_orient, self.pred_right_pose_params,
            self.pred_left_orient, self.pred_left_pose_params
        ]
        self.pred_pose_params = torch.cat(pose_params_list, dim=1)

        
    def __compute_loss(self, loss_weights):
        # joints 2d loss
        # calculated from gt, used for print log
        joints_2d_loss, _ = self.loss_util._joints_2d_loss(
            self.joints_2d[:, :, :2], self.pred_joints_2d, self.joints_2d[:, :, 2:3])
        self.joints_2d_loss = joints_2d_loss

        # calculated from prediction, used for bp
        self.joints_2d_loss_p, self.joints_2d_loss_p_batch = self.loss_util._joints_2d_loss(
            self.init_joints_2d[:, :, :2], self.pred_joints_2d, self.init_joints_2d[:, :, 2:3])
        self.joints_2d_loss_p *= loss_weights['joints_2d_loss']
        self.joints_2d_loss_p_batch *= loss_weights['joints_2d_loss']
        self.loss = self.joints_2d_loss_p

        # joints_3d loss 
        # calculated from gt, used for print log
        joints_3d_loss, _ = self.loss_util._joints_3d_loss(
            self.joints_3d[:, :, :3].clone(), self.pred_joints_3d, self.joints_3d[:, :, 3:4])
        self.joints_3d_loss = joints_3d_loss * 1000

        # calculated from prediction, used for bp
        self.joints_3d_loss_p, self.joints_3d_loss_p_batch = self.loss_util._joints_3d_loss(
            self.init_joints_3d[:, :, :3].clone(), self.pred_joints_3d, self.init_joints_3d[:, :, 3:4])
        self.joints_3d_loss_p *= loss_weights['joints_3d_loss']
        self.joints_3d_loss_p_batch *= loss_weights['joints_3d_loss']
        self.loss = (self.loss + self.joints_3d_loss_p)

        # hand translation loss 
        # calculated from gt, used for print log
        hand_trans_loss = self.loss_util._hand_trans_loss(
            self.hand_trans[:, :, :3], self.pred_hand_trans, self.hand_trans[:, :, 3:4])
        self.hand_trans_loss = hand_trans_loss * 10

        # calculated from prediction, used for bp
        hand_trans_loss_p = self.loss_util._hand_trans_loss(
            self.init_hand_trans_j[:, :, :3], self.pred_hand_trans, self.init_hand_trans_j[:, :, 3:4])
        self.hand_trans_loss_p = hand_trans_loss_p * loss_weights['trans_loss_weight']
        self.loss = (self.loss + self.hand_trans_loss_p)

        # collision loss
        collision_loss, self.collision_loss_batch, self.collision_loss_origin_scale = self.loss_util._collision_loss(
            self.pred_right_hand_verts, self.pred_left_hand_verts, self.hand_type_array)
        self.collision_loss = collision_loss * loss_weights['collision_loss_weight']
        self.loss = (self.loss + self.collision_loss)

        # shape reg loss
        pred_shape_params = torch.cat((self.pred_right_shape_params, self.pred_left_shape_params), dim=1)
        shape_reg_loss = self.loss_util._shape_reg_loss(pred_shape_params)
        self.shape_reg_loss = shape_reg_loss * loss_weights['shape_reg_loss_weight']
        self.loss = (self.loss + self.shape_reg_loss)

        # finger reg loss
        finger_reg_loss, _ = self.loss_util._finger_reg_loss(self.pred_joints_3d)
        self.finger_reg_loss = finger_reg_loss * loss_weights['finger_reg_loss_weight']
        self.loss = (self.loss + self.finger_reg_loss)


    def __set_optimize_target(self, stage):
        update_params = stage['update_params']
        opt_params = list()
        for param_name in update_params:
            assert param_name.startswith("pred_")
            assert hasattr(self, param_name), f"{param_name}"
            param = getattr(self, param_name)
            param.requires_grad = True
            opt_params.append(param)

        if self.opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(opt_params, lr=stage['lr'], betas=(0.9, 0.999))
        else:
            assert self.opt.optimizer == 'sgd'
            self.optimizer = torch.optim.SGD(opt_params, lr=stage['lr'], momentum=0.9)
    

    def __init_mid_results(self):
        self.mid_results = list()
    

    def __save_mid_results(self, stage):
        mid_result = dict()
        # save params
        update_params = stage['update_params']
        for param_name in update_params:
            assert hasattr(self, param_name), f"{param_name}"
            param = getattr(self, param_name)
            param = param.detach().clone()
            mid_result[param_name] = param
        # save loss
        cand_losses = [item[0] for item in stage['filter_loss']]
        cand_losses.append(stage['select_loss'])
        for loss_name in cand_losses:
            assert opt_utils.check_valid_loss(loss_name)
            loss_name_batch = f"{loss_name}_batch"
            assert hasattr(self, loss_name_batch), f"{loss_name}"
            loss = getattr(self, loss_name_batch)
            loss = loss.detach().clone()
            mid_result[loss_name] = loss
        # add to overall
        self.mid_results.append(mid_result)
    

    def __update_stage_results(self, stage):
        # gather all params
        all_params, all_losses = opt_utils.gather_params_losses(
            self.mid_results, stage)
        # filter params 
        updated_losses = opt_utils.filter_by_losses(all_losses, stage['filter_loss'])
        # select params
        select_params = opt_utils.select_params(
            all_params, updated_losses, stage['select_loss'])
        for param_name in select_params:
            setattr(self, param_name, select_params[param_name])


    def optimize(self, iter_id, num_iter):
        num_stage = len(self.strategy)

        for stage_id, stage in enumerate(self.strategy):
            self.__set_optimize_target(stage) # set-up params to be updated in this stage.
            self.__init_mid_results() # init variable for storing the middle optimize results.
            # run opt
            epoch = stage['epoch']
            for j in range(epoch+1): # +1 to make sure epoch is valid.
                self.forward()
                self.__compute_loss(stage['loss_weights'])
                # save middle results here
                if j % self.opt.save_mid_freq == 0:
                    self.__save_mid_results(stage)
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
            self.__update_stage_results(stage)
            if self.process_rank <= 0:
                print(f"iter:{iter_id+1:04d}/{num_iter:04d}, stage-{stage_id:02d} completes")
                sys.stdout.flush()
        
        # after optimization complete, forward again
        self.forward()
        self.__compute_loss(self.default_loss_weights)
        # print('final', self.joints_3d_loss, self.joints_3d_loss_p)


    def get_pred_result(self):

        pred_result = OrderedDict(
            pred_cam_params = self.pred_cam_params.detach().cpu().numpy(),
            pred_hand_trans = self.pred_hand_trans.detach().cpu().numpy(),
            pred_shape_params = self.pred_shape_params.detach().cpu().numpy(),
            pred_pose_params = self.pred_pose_params.detach().cpu().numpy(),
            pred_right_hand_verts = self.pred_right_hand_verts.detach().cpu().numpy(),
            pred_left_hand_verts = self.pred_left_hand_verts.detach().cpu().numpy(),
            mano_params_weight = self.mano_params_weight.detach().cpu().numpy(),
            pred_joints_3d = self.pred_joints_3d.detach().cpu().numpy(),
            gt_joints_3d = self.joints_3d.cpu().numpy(),
            collision_loss = self.collision_loss_batch.detach().cpu().numpy(),
            collision_loss_origin_scale = self.collision_loss_origin_scale.detach().cpu().numpy(),
            do_flip = np.zeros(self.batch_size).astype(np.int32),
            pred_hand_type = np.ones(self.batch_size).astype(np.int32),
        )
        return pred_result


    def get_current_errors(self):
        joints_2d_loss = self.joints_2d_loss.item()
        loss_dict = OrderedDict([('joints_2d_loss', joints_2d_loss)])

        joints_3d_loss = self.joints_3d_loss.item()
        loss_dict['joints_3d_loss'] = joints_3d_loss

        hand_trans_loss = self.hand_trans_loss.item()
        loss_dict['hand_trans_loss'] = hand_trans_loss

        collision_loss = self.collision_loss.item()
        loss_dict['collision_loss'] = collision_loss

        loss_dict['joints_3d_loss_p'] = self.joints_3d_loss_p.item()
        # loss_dict['joints_2d_loss_p'] = self.joints_2d_loss_p.item()
        # loss_dict['shape_reg'] = self.shape_reg_loss.item()

        return loss_dict