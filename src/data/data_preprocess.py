import os
import sys
import shutil
import os.path as osp
import random
from datetime import datetime
import numpy as np
import torchvision.transforms as transforms
import torch
import cv2
from PIL import Image
import utils.rotate_utils as ru
import utils.geometry_utils as gu
from data import data_utils


class DataProcessor(object):
    def __init__(self, opt):
        self.opt = opt
        self.rescale_range = [0.6, 1.0]
        self.angle_scale = [-90, 90]
        self.num_slice = 10
        self.color_transfomer = transforms.ColorJitter(
            brightness = (0.9, 1.3),
            contrast = (0.8, 1.3),
            saturation = (0.4, 1.6),
            hue = (-0.1, 0.1)
        )

        if opt.model_type == 'baseline' and opt.isTrain and self.opt.use_motion_blur:
            self.blur_kernels = data_utils.load_blur_kernel(opt.blur_kernel_dir) 
            self.motion_blur_prob = opt.motion_blur_prob
    

    def hand_type_str2array(self, hand_type):
        if hand_type == 'right':
            return np.array([1, 0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0, 1], dtype=np.float32)
        else:
            assert hand_type == 'interacting', f"{hand_type} not supported."
            return np.array([1, 1], dtype=np.float32)


    def padding_and_resize(self, img, joints_2d):
        final_size = self.opt.inputSize
        height, width = img.shape[:2]
        if height > width:
            ratio = final_size / height
            new_height = final_size
            new_width = int(ratio * width)
        else:
            ratio = final_size / width
            new_width = final_size
            new_height = int(ratio * height)
        new_img = np.zeros((final_size, final_size, 3), dtype=np.uint8)
        new_img[:new_height, :new_width, :] = cv2.resize(img, (new_width, new_height))

        joints_2d[:, :2] *= ratio
        return new_img, joints_2d
    

    def random_flip(self, img, hand_type_array, joints_2d, joints_3d, mano_params, do_flip=False):
        if np.random.random() > 0.5 or do_flip:
            # img
            img_new = np.fliplr(img).copy()
            # hand type
            hand_type_array_new = np.flip(hand_type_array).copy()
            # joints 2d
            joints_2d_new = np.zeros((42, 3), dtype=np.float32)
            joints_2d_new[:21, :] = joints_2d[21:, :].copy()
            joints_2d_new[21:, :] = joints_2d[:21, :].copy()
            joints_2d_new[:, 0] = img.shape[1] - joints_2d_new[:, 0]
            # joints 3d
            joints_3d_new = np.zeros((42, 4), dtype=np.float32)
            joints_3d_new[:21, :] = joints_3d[21:, :].copy()
            joints_3d_new[21:, :] = joints_3d[:21, :].copy()
            joints_3d_new[:, 0] = -joints_3d_new[:, 0]
            # mano params
            mano_pose, mano_betas, mano_params_weight = mano_params
            mano_pose_new = np.zeros((48*2), dtype=np.float32)
            mano_betas_new = np.zeros((10*2), dtype=np.float32)
            mano_params_weight_new = np.zeros((2), dtype=np.float32)
            mano_pose_new[:48] = gu.flip_hand_pose(mano_pose[48:])
            mano_pose_new[48:] = gu.flip_hand_pose(mano_pose[:48])
            mano_betas[:10] = mano_betas[10:]
            mano_betas[10:] = mano_betas[:10]
            mano_params_weight_new[0] = mano_params_weight[1]
            mano_params_weight_new[1] = mano_params_weight[0]
            mano_params_new = (mano_pose_new, mano_betas_new, mano_params_weight_new)
            return img_new, hand_type_array_new, joints_2d_new, joints_3d_new, mano_params_new, True
        else:
            return img, hand_type_array, joints_2d, joints_3d, mano_params, False


    def random_rescale(self, img, joints_2d, use_random_position=False):
        # resize
        min_s, max_s = self.rescale_range
        final_size = self.opt.inputSize
        random_scale = random.random() * (max_s-min_s) + min_s
        new_size = int(final_size * random_scale)
        res_img = np.zeros((final_size, final_size, 3), dtype=np.uint8)
        # pose
        y_pos, x_pos = 0, 0
        if use_random_position:
            height, width = img.shape[:2]
            assert height==width
            end = final_size-new_size-1
            x_pos = random.randint(0, end)
            y_pos = random.randint(0, end)

        new_img = cv2.resize(img, (new_size, new_size))
        res_img[y_pos:new_size+y_pos, x_pos:new_size+x_pos, :] = new_img

        joints_2d[:, :2] *= random_scale
        joints_2d[:, 0] += x_pos
        joints_2d[:, 1] += y_pos

        return res_img, joints_2d
    

    def random_rotate(self, img, joints_2d, joints_3d, mano_pose):
        min_angle, max_angle = self.angle_scale
        num_slice = self.num_slice
        slice_id = random.randint(0, num_slice-1)
        angle = (max_angle-min_angle)/num_slice * slice_id + min_angle
        # image
        img = ru.rotate_image(img.copy(), angle)
        # orient of hand
        rot_orient = ru.rotate_orient(mano_pose[:3], angle)
        mano_pose[:3] = rot_orient
        # joints 2d
        origin = np.array((img.shape[1]/2, img.shape[0]/2)).reshape(1,2)
        joints_2d_valid = joints_2d[:, -1:]
        joints_2d = joints_2d[:, :2]
        joints_2d = ru.rotate_joints_2d(joints_2d, origin, angle)
        joints_2d = np.concatenate((joints_2d, joints_2d_valid), axis=1)
        # joints 3d
        joints_3d_valid = joints_3d[:, -1:]
        joints_3d = joints_3d[:, :3]
        joints_3d = ru.rotate_joints_3d(joints_3d.T, angle)
        joints_3d = np.concatenate((joints_3d, joints_3d_valid), axis=1)
        return img, joints_2d, joints_3d, mano_pose


    def color_jitter(self, img):
        pil_img = Image.fromarray(img)
        transformed_img = self.color_transfomer(pil_img)
        tmp_img = np.asarray(transformed_img)
        res_img = np.zeros((tmp_img.shape), dtype=np.uint8)
        res_img[:] = tmp_img[:]
        return res_img
    

    def add_motion_blur(self, img):
        if random.random() < self.motion_blur_prob:
            blur_kernel = random.choice(self.blur_kernels)
            img = cv2.filter2D(img, -1, blur_kernel)
        return img


    def normalize_joints_2d(self, joints_2d):
        final_size = self.opt.inputSize

        joints_2d_new = np.copy(joints_2d)
        joints_2d_new[:, 0] = (joints_2d[:, 0] / final_size) * 2.0 - 1.0
        joints_2d_new[:, 1] = (joints_2d[:, 1] / final_size) * 2.0 - 1.0

        return joints_2d_new