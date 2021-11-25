
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
from .networks import InterHandEncoder
from .transform_utils import batch_orthogonal_project
from .loss_utils import LossUtil
import time
from utils import vis_util
from utils import render_color_utils as rcu
import ry_utils


class InterHandModel(BaseModel):
    @property
    def name(self):
        return 'InterHandModel'

    def __init__(self, opt):
        BaseModel.initialize(self, opt)

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

        if opt.isTrain:
            self.use_collision_loss = opt.use_collision_loss
        else:
            self.use_collision_loss = True

        # initialize inputs
        self.init_input()

        # load mean params, the mean params are from HMR
        self.load_mean_params()

        # load mano models
        self.load_mano_model()

        # loss utils
        self.loss_util = LossUtil(opt, self.mano_models)

        # set encoder and optimizer
        self.encoder = InterHandEncoder(opt, self.mean_params).cuda()
        if opt.dist:
            self.encoder = DistributedDataParallel(
                self.encoder, device_ids=[torch.cuda.current_device()])
        if self.isTrain:
            self.optimizer = torch.optim.Adam(
                self.encoder.parameters(), lr=opt.lr)
        
        # load pretrained / trained weights for encoder
        if self.isTrain:
            if opt.continue_train:
                # resume training from saved weights
                which_epoch = opt.which_epoch
                saved_info = self.load_info(which_epoch)
                opt.epoch_count = saved_info['epoch']
                self.optimizer.load_state_dict(saved_info['optimizer'])
                which_epoch = opt.which_epoch
                self.load_network(self.encoder, 'baseline', which_epoch)
                if opt.process_rank <= 0:
                    print('resume from epoch {}'.format(opt.epoch_count))
            else:
                if opt.pretrain_weights is None:
                    print("Alert: No pretrained weights !!!!!!!!!!!!!")
                    time.sleep(3) 
                else:
                    assert(osp.exists(opt.pretrain_weights))
                    if not self.opt.dist or self.opt.process_rank <= 0:
                        print("Load pretrained weights from {}".format(
                            opt.pretrain_weights))
                    if opt.dist:
                        self.encoder.module.load_state_dict(
                            torch.load(opt.pretrain_weights, 
                            map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())), 
                            strict=False)
                    else:
                        self.encoder.load_state_dict(torch.load(opt.pretrain_weights), strict=False)
        else:
            pass
    

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
                use_pca=False, is_rhand=is_rhand, batch_size=self.batch_size)
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
        self.input_img = self.Tensor(
            nb, self.opt.input_nc, self.inputSize, self.inputSize)
        # hand class
        self.hand_type_array = self.Tensor(nb, 2)
        self.hand_type_valid = self.Tensor(nb, 1)
        # joints 2d 
        self.joints_2d = self.Tensor(nb, self.opt.num_joints, 3)
        # joints 3d
        self.joints_3d = self.Tensor(nb, self.opt.num_joints, 4)
        # mano pose params
        self.gt_pose_params = self.Tensor(nb, self.opt.pose_params_dim)
        self.gt_shape_params = self.Tensor(nb, self.opt.shape_params_dim)
        self.mano_params_weight = self.Tensor(nb, 2)
        # set hand translation
        self.hand_trans = self.Tensor(nb, 1, 4)
        # do-flip
        self.do_flip = self.Tensor(nb,)


    def set_input(self, input):
        # image
        input_img = input['img']
        self.input_img.resize_(input_img.size()).copy_(input_img)
        # flip information
        do_flip = input['do_flip']
        self.do_flip.resize_(do_flip.size()).copy_(do_flip).bool()
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
        

    def get_mano_output(self, pose_params, shape_params, hand_trans, hand_type_array):
        output_verts = dict()
        output_joints = dict()

        for hand_type in ['right', 'left']:
            pose_shift = 0 if hand_type == 'right' else 48
            betas_shift = 0 if hand_type == 'right' else 10

            hand_rotation = pose_params[:, pose_shift:pose_shift+48][:, :3]
            hand_pose = pose_params[:, pose_shift:pose_shift+48][:, 3:]
            hand_betas = shape_params[:, betas_shift:betas_shift+10]

            output = self.mano_models[hand_type](
                global_orient = hand_rotation,
                hand_pose = hand_pose,
                betas = hand_betas)
            vertices = output.vertices
            joints_origin = output.joints # original joints
            joints = joints_origin

            J_regressor = self.mano_models[hand_type].J_regressor
            joints_new = torch.einsum('bik,ji->bjk', [vertices, J_regressor]) # new regressed joints

            extra_joints = torch.index_select(vertices, 1, self.joint_ids)
            joints = torch.cat([joints, extra_joints], dim=1)
            output_verts[hand_type] = vertices
            output_joints[hand_type] = joints
        
        # apply trans and merge joints
        left_hand_verts = output_verts['left']
        right_hand_verts = output_verts['right']
        left_hand_joints = output_joints['left']
        right_hand_joints = output_joints['right']

        # move left hand to right wrist
        left_to_right_shift = right_hand_joints[:, 0:1, :] - left_hand_joints[:, 0:1, :] 
        # left_to_right_shift = left_to_right_shift * (hand_type_array[:, 0:1]>0.5).view(self.batch_size, 1, 1)
        # apply predicted translation
        hand_trans = hand_trans.view(self.batch_size, 1, 3) 
        shift = hand_trans + left_to_right_shift

        left_hand_verts = left_hand_verts + shift
        left_hand_joints = left_hand_joints + shift
        # merge
        joints = torch.cat([right_hand_joints, left_hand_joints], dim=1) # (bs, 42, 3)

        return right_hand_verts, left_hand_verts, joints


    def forward(self):
        # get predicted params first
        self.final_params, self.pred_hand_type = self.encoder(self.input_img)

        # get predicted params for cam, pose, shape
        cam_dim = self.cam_params_dim
        pose_dim = self.pose_params_dim
        shape_dim = self.shape_params_dim
        self.pred_cam_params = self.final_params[:, :cam_dim]
        self.pred_pose_params = self.final_params[:, cam_dim: 
            (cam_dim + pose_dim)]
        self.pred_shape_params = self.final_params[:, (cam_dim + pose_dim): 
            (cam_dim + pose_dim + shape_dim)]
        self.pred_hand_trans = self.final_params[:, (cam_dim + pose_dim + shape_dim):]

        # get pred verts and joints 3d
        self.pred_right_hand_verts, self.pred_left_hand_verts, self.pred_joints_3d = self.get_mano_output(
            self.pred_pose_params, self.pred_shape_params, self.pred_hand_trans, self.hand_type_array)

        # generate predicted joints 2d
        self.pred_joints_2d = batch_orthogonal_project(
            self.pred_joints_3d, self.pred_cam_params)
        
        # get gt verts and joints 3d
        self.gt_right_hand_verts, self.gt_left_hand_verts, self.gt_joints_3d_mano = self.get_mano_output(
            self.gt_pose_params, self.gt_shape_params, self.hand_trans[:, :, :3], self.hand_type_array)


    def backward_E(self):
        # hand class loss (handedness)
        self.hand_type_loss = self.loss_util._hand_type_loss(
            self.hand_type_array, self.pred_hand_type, self.hand_type_valid)
        self.loss = self.hand_type_loss

        # joints 2d loss
        self.joints_2d_loss, _ = self.loss_util._joints_2d_loss(
            self.joints_2d[:, :, :2], self.pred_joints_2d, self.joints_2d[:, :, 2:3])
        self.joints_2d_loss *= self.opt.joints_2d_loss_weight
        self.loss = (self.loss + self.joints_2d_loss)

        # joints_3d loss
        self.joints_3d_loss, _ = self.loss_util._joints_3d_loss(
            self.joints_3d[:, :, :3], self.pred_joints_3d, self.joints_3d[:, :, 3:4])
        self.joints_3d_loss *= self.opt.joints_3d_loss_weight
        self.loss = (self.loss + self.joints_3d_loss)

        # mano pose loss
        mano_right_pose_loss = self.loss_util._mano_pose_loss(self.gt_pose_params[:, :48], 
            self.pred_pose_params[:, :48], self.mano_params_weight[:, 0:1])
        mano_left_pose_loss = self.loss_util._mano_pose_loss(self.gt_pose_params[:, 48:], 
            self.pred_pose_params[:, 48:], self.mano_params_weight[:, 1:2])
        self.mano_pose_loss = mano_right_pose_loss + mano_left_pose_loss
        self.mano_pose_loss = self.mano_pose_loss * self.opt.pose_param_weight
        self.loss = (self.loss + self.mano_pose_loss)

        # mano shape loss
        mano_right_shape_loss = self.loss_util._mano_shape_loss(self.gt_shape_params[:, :10], 
            self.pred_shape_params[:, :10], self.mano_params_weight[:, 0:1])
        mano_left_shape_loss = self.loss_util._mano_shape_loss(self.gt_shape_params[:, 10:], 
            self.pred_shape_params[:, 10:], self.mano_params_weight[:, 1:2])
        self.mano_shape_loss = mano_right_shape_loss + mano_left_shape_loss
        self.mano_shape_loss = self.mano_shape_loss * self.opt.shape_param_weight
        self.loss = (self.loss + self.mano_shape_loss)

        # hand translation loss
        self.hand_trans_loss, _ = self.loss_util._hand_trans_loss(
            self.hand_trans[:, :, :3], self.pred_hand_trans, self.hand_trans[:, :, 3:4])
        self.hand_trans_loss = self.hand_trans_loss * self.opt.trans_loss_weight
        self.loss = (self.loss + self.hand_trans_loss)
    
        # shape constrain loss
        self.shape_reg_loss = self.loss_util._shape_reg_loss(self.pred_shape_params)
        self.shape_reg_loss = self.shape_reg_loss * self.opt.shape_reg_loss_weight
        self.loss = (self.loss + self.shape_reg_loss)

        # collision loss
        # In practice, collisino loss is not used in baseline model training
        if self.use_collision_loss:
            self.collision_loss, _, _ = self.loss_util._collision_loss(
                self.pred_right_hand_verts, self.pred_left_hand_verts, self.hand_type_array)
            self.collision_loss = self.collision_loss * self.opt.collision_loss_weight
            self.loss = (self.loss + self.collision_loss)

        # backward
        self.loss.backward()


    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.backward_E()
        self.optimizer.step()


    def test(self):
        with torch.no_grad():
            self.forward()
            # calculate collision loss for evaluation
            _, _, self.collision_loss_origin_scale = self.loss_util._collision_loss(
                self.pred_right_hand_verts, self.pred_left_hand_verts, self.hand_type_array)


    def get_pred_result(self):
        pred_result = OrderedDict(
            pred_cam_params = self.pred_cam_params.cpu().numpy(),
            pred_hand_type = self.pred_hand_type.cpu().numpy(),
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
            collision_loss_origin_scale = self.collision_loss_origin_scale.cpu().numpy(),
            do_flip = self.do_flip.cpu().numpy(),
        )
        return pred_result


    def get_current_errors(self):
        joints_2d_loss = self.joints_2d_loss.item()
        loss_dict = OrderedDict([('joints_2d_loss', joints_2d_loss)])

        hand_type_loss = self.hand_type_loss.item()
        loss_dict['hand_type_loss'] = hand_type_loss
        
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

        if self.use_collision_loss:
            collision_loss = self.collision_loss.item()
            loss_dict['collision_loss'] = collision_loss
        else:
            loss_dict['collision_loss'] = 0.0

        total_loss = self.loss.item()
        loss_dict['total_loss'] = total_loss

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


    def save(self, label, epoch):
        self.save_network(self.encoder, 'baseline', label)
        save_info = {'epoch': epoch,
                     'optimizer': self.optimizer.state_dict()}
        self.save_info(save_info, label)


    def eval(self):
        self.encoder.eval()

    def update_learning_rate(self, epoch):
        if self.opt.lr_decay_type == 'cosine':
            old_lr = self.opt.lr
            lr = 0.5*(1.0 + np.cos(np.pi*epoch/self.opt.total_epoch)) * old_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        elif self.opt.lr_decay_type == 'stage':
            assert self.opt.total_epoch == 20, f"Stage strategy is only supports total epoch equals to 20"
            if epoch in [15, 17]:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] / 10
            lr = self.optimizer.param_groups[0]['lr']
        else:
            assert self.opt.lr_decay_type == 'none' # do nothing
            lr = self.opt.lr

        if self.opt.process_rank <= 0:
            print("Current Learning Rate:{0:.2E}".format(lr))
