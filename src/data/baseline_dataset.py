import os, sys, shutil
import os.path as osp
import random
from datetime import datetime
import numpy as np
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
from PIL import Image
import cv2
import pickle
from data.data_preprocess import DataProcessor
from data import data_utils
from utils.vis_util import draw_keypoints
import ry_utils

class BaselineDataset(data.Dataset):

    def __init__(self, opt, dataset_info):
        super(BaselineDataset, self).__init__()

        name, anno_path, image_root = dataset_info
        self.name = name
        self.anno_path = anno_path
        # self.image_root = image_root
        self.image_root = osp.join(opt.data_root, image_root)

        self.opt = opt
        self.isTrain = opt.isTrain
        self.data_root = opt.data_root
        self.param_root = opt.param_root
        self.data_processor = DataProcessor(opt)
        if self.isTrain:
            self.use_random_flip = opt.use_random_flip
            self.use_random_rescale = opt.use_random_rescale
            self.use_random_position = opt.use_random_position
            self.use_random_rotation = opt.use_random_rotation
            self.use_color_jittering = opt.use_color_jittering
            self.use_motion_blur = opt.use_motion_blur

        transform_list = [ transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)
    

    def load_data(self):
        data_list = data_utils.load_annotation(self.data_root, self.anno_path)
        data_list = sorted(data_list, key=lambda a:a['img_path'])
        self.all_data_list = data_list

        if self.isTrain:
            self.data_list = data_list

        if self.isTrain:
            self.num_add = 0 # there is no necessary to pad data in training
        else:
            if self.opt.dist:
                # the total number of data should be divisible by bs * num_process
                bs = self.opt.batchSize * torch.distributed.get_world_size() 
            else:
                bs = self.opt.batchSize
            self.num_add = bs - len(data_list) % bs
            self.data_list = data_list + data_list[0:1] * self.num_add
    

    def preprocess_data(self, img, hand_type_array, joints_2d, joints_3d, mano_params):
        # pad and resize, 
        img, joints_2d = self.data_processor.padding_and_resize(img, joints_2d)

        if hand_type_array[0]<0.5 and hand_type_array[1]>0.5: 
            # flip left-hand to right, no matter train or test
            img, hand_type_array, joints_2d, joints_3d, mano_params, do_flip = self.data_processor.random_flip( 
                img, hand_type_array, joints_2d, joints_3d, mano_params, do_flip=True)
        elif self.isTrain and self.use_random_flip and np.sum(hand_type_array)>1.5:
            # flip inter-hand only in training
            img, hand_type_array, joints_2d, joints_3d, mano_params, do_flip = self.data_processor.random_flip( 
                img, hand_type_array, joints_2d, joints_3d, mano_params)
        else:
            do_flip = False

        # unpack mano params
        mano_pose, mano_betas, mano_params_weight = mano_params
        
        # random scale
        if self.isTrain and self.use_random_rescale:
            img, joints_2d = self.data_processor.random_rescale(
                img, joints_2d, self.use_random_position)

        # random rotation
        if self.isTrain and self.use_random_rotation: 
            img, joints_2d, joints_3d, mano_pose = self.data_processor.random_rotate(
                img, joints_2d, joints_3d, mano_pose)

        # random color jittering
        if self.isTrain and self.use_color_jittering:
            img = self.data_processor.color_jitter(img)
        
        # motion blur augmentation
        if self.isTrain and self.use_motion_blur:
            img = self.data_processor.add_motion_blur(img)

        mano_params = (mano_pose, mano_betas, mano_params_weight)

        # normalize coords of keypoinst to [-1, 1]
        joints_2d = self.data_processor.normalize_joints_2d(joints_2d)

        return img, hand_type_array, joints_2d, joints_3d, mano_params, do_flip
    

    def __getitem__(self, index):
        # load raw data
        single_data = self.data_list[index]
        if 'param_path' in single_data: 
            # for hand26m only
            param_path = osp.join(self.param_root, single_data['param_path'])
            param_data = ry_utils.load_pkl(param_path)
            single_data = {**single_data, **param_data} # merge two dict

        # image
        img_path = single_data['img_path'] # relative path
        img_path_full = osp.join(self.image_root, img_path) # full path
        img = cv2.imread(img_path_full)
        ori_img_size = np.max(img.shape[:2])

        # hand type 
        if 'hand_type' in single_data:
            hand_type = single_data['hand_type']
        else:
            hand_type = 'interacting'
        hand_type_array = self.data_processor.hand_type_str2array(hand_type)
        # hand_type valid
        hand_type_valid = np.ones((1, ), dtype=np.float32)
        if 'hand_type_valid' in single_data:
            hand_type_valid[0] = single_data['hand_type_valid']

        # joints 2D
        if self.isTrain:
            assert "joints_2d" in single_data, "Joints 2D must be provided by training data"
        if "joints_2d" in single_data:
            joints_2d = single_data['joints_2d'].copy()
        else:
            joints_2d = np.zeros((self.opt.num_joints, 3))
        if joints_2d.shape[1] == 2:
            num_joints = joints_2d.shape[0]
            score = np.ones((num_joints, 1), dtype=np.float32)
            joints_2d = np.concatenate((joints_2d, score), axis=1)
        
        # joints 3D
        if "joints_3d" in single_data:
            joints_3d = single_data['joints_3d'].copy()
        else:
            joints_3d = np.zeros((self.opt.num_joints, 3))
        if joints_3d.shape[1] == 3:
            num_joints = joints_3d.shape[0]
            score = np.ones((num_joints, 1), dtype=np.float32)
            joints_3d = np.concatenate((joints_3d, score), axis=1)
        # scale ratio of joints 3d
        if "scale" in single_data:
            scale_ratio = single_data['scale']
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
            if key in single_data:
                value = single_data[key]
                if value is not None:
                    pose = value['pose']
                    betas = value['shape']
                    mano_pose[pose_shift : pose_shift+48] = pose
                    mano_betas[betas_shift : betas_shift+10] = betas
                    mano_params_weight[weights_idx] = 1
        # pack mano params
        mano_params = (mano_pose, mano_betas, mano_params_weight)

        # preprocess the images and the corresponding annotation
        img, hand_type_array, joints_2d, joints_3d, mano_params, do_flip = self.preprocess_data(
            img, hand_type_array, joints_2d, joints_3d, mano_params)

        # unpack processed mano params
        mano_pose, mano_betas, mano_params_weight = mano_params

        # translation
        if joints_3d[0, -1]>0.0 and joints_3d[21, -1]>0.0:
            hand_trans = -joints_3d[0, :3] + joints_3d[21, :3]
            hand_trans_weight = np.ones((1,), dtype=np.float32)
        else:
            hand_trans = np.zeros((3,), dtype=np.float32)
            hand_trans_weight = np.zeros((1, ), dtype=np.float32)
        hand_trans = np.concatenate((
            hand_trans, hand_trans_weight)).reshape(1, 4)

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

        result = dict(
            img = img,
            joints_2d = joints_2d,
            joints_3d = joints_3d,
            mano_pose = mano_pose,
            mano_betas = mano_betas,
            mano_params_weight = mano_params_weight,
            hand_trans = hand_trans,
            hand_type_array = hand_type_array,
            hand_type_valid = hand_type_valid,
            do_flip = torch.tensor(do_flip),
            scale_ratio = torch.tensor(scale_ratio),
            ori_img_size = torch.tensor(ori_img_size),
            index = torch.tensor(index),
        )
        return result


    def getitem(self, index):
        return self.__getitem__(index)

    
    def __len__(self):
        return len(self.data_list)
