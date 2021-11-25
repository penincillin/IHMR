import os, sys, shutil
import os.path as osp
import random
from datetime import datetime
import numpy as np
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
import cv2
import pdb
from data.data_preprocess import DataProcessor
from data import data_utils
from utils.vis_util import draw_keypoints
import ry_utils

class MLPDataset(data.Dataset):

    def __init__(self, opt, dataset_info):
        super(MLPDataset, self).__init__()

        name, anno_path, pred_res_path, image_root = dataset_info
        self.name = name
        self.anno_path = anno_path
        self.pred_res_path = pred_res_path
        self.image_root = osp.join(opt.data_root, image_root)

        self.opt = opt
        self.data_root = opt.data_root
        self.param_root = opt.param_root
        self.use_opt_params = opt.use_opt_params
        self.data_processor = DataProcessor(opt)

        transform_list = [ transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)
    

    def load_data(self):
        # load data from both annotation and init prediction
        data_list = data_utils.load_anno_pred_data(
            self.data_root, self.anno_path, self.pred_res_path)

        self.num_add = 0
        if self.opt.dist:
            # the total number of data should be divisible by bs * num_process
            bs = self.opt.batchSize * torch.distributed.get_world_size() 
        else:
            bs = self.opt.batchSize
        num_add = bs - len(data_list) % bs
        self.num_add = 0 if num_add == bs else num_add
        self.data_list = data_list + data_list[0:1] * self.num_add
    

    def preprocess_data(self, img, joints_2d):
        # pad and resize, 
        img, joints_2d = self.data_processor.padding_and_resize(img, joints_2d)
        # normalize coords of keypoinst to [-1, 1]
        joints_2d = self.data_processor.normalize_joints_2d(joints_2d)
        # return
        return img, joints_2d
    

    def __getitem__(self, index):
        # load anno data
        anno_data = self.data_list[index]
        param_path = osp.join(self.param_root, anno_data['param_path'])
        param_data = ry_utils.load_pkl(param_path)
        anno_data = {**anno_data, **param_data} # merge two dict

        # image
        img_path = anno_data['img_path'] # relative path
        img_path_full = osp.join(self.image_root, img_path) # full path
        img = cv2.imread(img_path_full)

        # hand type && hand type valid
        hand_type = anno_data['hand_type']
        hand_type_array = self.data_processor.hand_type_str2array(hand_type)
        hand_type_valid = np.zeros((1, ), dtype=np.float32)
        hand_type_valid[0] = anno_data['hand_type_valid']

        # joints 2D (to be updated, should use predicted 2D joints)
        if "joints_2d" in anno_data:
            joints_2d = anno_data['joints_2d'].copy()
        else:
            joints_2d = np.zeros((self.opt.num_joints, 3))
        if joints_2d.shape[1] == 2:
            num_joints = joints_2d.shape[0]
            score = np.ones((num_joints, 1), dtype=np.float32)
            joints_2d = np.concatenate((joints_2d, score), axis=1)
        
        # joints 3D
        if "joints_3d" in anno_data:
            joints_3d = anno_data['joints_3d'].copy()
        else:
            joints_3d = np.zeros((self.opt.num_joints, 3))
        if joints_3d.shape[1] == 3:
            num_joints = joints_3d.shape[0]
            score = np.ones((num_joints, 1), dtype=np.float32)
            joints_3d = np.concatenate((joints_3d, score), axis=1)
        # scale ratio of joints 3d
        if "scale" in anno_data:
            scale_ratio = anno_data['scale']
        else:
            scale_ratio = 1.0
        
        # mano params
        mano_pose = np.zeros((48*2,), dtype=np.float32) # (96, )
        mano_betas = np.zeros((10*2,), dtype=np.float32) # (20, )
        mano_params_weight = np.zeros((2,), dtype=np.float32) # (2, )
        for hand_type in ['right', 'left']:
            pose_shift = (0 if hand_type == 'right' else 48)
            betas_shift = (0 if hand_type == 'right' else 10)
            weights_idx = (0 if hand_type == 'right' else 1)
            key = f'{hand_type}_hand_param'
            value = anno_data[key]
            if value is not None and not self.opt.use_opt_params:
                pose = value['pose']
                betas = value['shape']
                mano_pose[pose_shift : pose_shift+48] = pose
                mano_betas[betas_shift : betas_shift+10] = betas
                mano_params_weight[weights_idx] = 1
            else:
                pred_pose_params = anno_data['pose_params_opt']
                pred_shape_params = anno_data['shape_params_opt']
                mano_pose[pose_shift: pose_shift+48] = \
                    pred_pose_params[pose_shift: pose_shift+48]
                mano_betas[betas_shift : betas_shift+10] = \
                    pred_shape_params[betas_shift:betas_shift+10]
                mano_params_weight[weights_idx] = 1

        # translation
        if joints_3d[0, -1]>0.0 and joints_3d[21, -1]>0.0:
            hand_trans = -joints_3d[0, :3] + joints_3d[21, :3]
            hand_trans_weight = np.ones((1,), dtype=np.float32)
        else:
            hand_trans = np.zeros((3,), dtype=np.float32)
            hand_trans_weight = np.zeros((1, ), dtype=np.float32)
        hand_trans = np.concatenate((
            hand_trans, hand_trans_weight)).reshape(1, 4)
        
        if self.use_opt_params:
            hand_trans_opt = anno_data['hand_trans_opt']
            hand_trans[:, :3] = hand_trans_opt
            hand_trans[:, 3] = 1.0

        # Init predictions
        init_cam = anno_data['pred_cam_params']
        init_shape_params = anno_data['pred_shape_params']
        init_pose_params = anno_data['pred_pose_params']
        init_hand_trans = anno_data['pred_hand_trans']

        # joints
        init_joints_2d = anno_data['pred_joints_2d']
        init_joints_3d = anno_data['pred_joints_3d']
        score = np.ones((init_joints_2d.shape[0], 1))
        init_joints_2d = np.concatenate((init_joints_2d, score), axis=1)
        init_joints_3d = np.concatenate((init_joints_3d, score), axis=1)

        # preprocess the images and the corresponding annotation
        img_origin = img
        img, joints_2d = self.preprocess_data(img_origin.copy(), joints_2d)
        _, init_joints_2d = self.preprocess_data(img_origin.copy(), init_joints_2d)

        # transfer data from numpy to torch tensor
        img = self.transform(img).float()
        joints_2d = torch.from_numpy(joints_2d).float()
        mano_pose = torch.from_numpy(mano_pose).float()
        mano_betas = torch.from_numpy(mano_betas).float()
        mano_params_weight = torch.from_numpy(mano_params_weight).float()
        joints_3d = torch.from_numpy(joints_3d).float()
        hand_trans = torch.from_numpy(hand_trans).float()
        hand_type_array = torch.from_numpy(hand_type_array).float()
        hand_type_valid = torch.from_numpy(hand_type_valid).float()
        # init params
        init_cam = torch.from_numpy(init_cam).float()
        init_shape_params = torch.from_numpy(init_shape_params).float()
        init_pose_params = torch.from_numpy(init_pose_params).float()
        init_hand_trans = torch.from_numpy(init_hand_trans).float()
        init_joints_2d = torch.from_numpy(init_joints_2d).float()
        init_joints_3d = torch.from_numpy(init_joints_3d).float()
        # image feature
        img_feat = torch.from_numpy(anno_data['img_feat']).float()

        result = dict(
            img = img,
            # gt
            joints_2d = joints_2d,
            joints_3d = joints_3d,
            mano_pose = mano_pose,
            mano_betas = mano_betas,
            mano_params_weight = mano_params_weight,
            hand_trans = hand_trans,
            hand_type_array = hand_type_array,
            hand_type_valid = hand_type_valid,
            scale_ratio = torch.tensor(scale_ratio),
            index = torch.tensor(index),
            # init params
            init_cam = init_cam,
            init_shape_params = init_shape_params,
            init_pose_params = init_pose_params,
            init_hand_trans = init_hand_trans,
            init_joints_2d = init_joints_2d,
            init_joints_3d = init_joints_3d,
            # init_hand_trans_j = init_hand_trans_j,
            # img feat
            img_feat = img_feat,
        )

        return result


    def getitem(self, index):
        return self.__getitem__(index)

    
    def __len__(self):
        return len(self.data_list)
