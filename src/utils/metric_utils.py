import os, sys, shutil
import os.path as osp
import cv2
import numpy as np
import time
import ry_utils
import numpy as np
import pdb
from collections import defaultdict


def get_hand_type_acc(hand_type, hand_type_valid, pred_hand_type):
    if hand_type_valid > 0:
        if hand_type == 'interacting':
            result = pred_hand_type[0]>0.5 and pred_hand_type[1]>0.5
        else:
            result = pred_hand_type[0]>0.5 and pred_hand_type[1]<0.5 # check right hand only, since left -> right
        return [result,]
    else:
        return []


def get_single_joints_error(joints_3d_1, joints_3d_2, joint_weights, scale_factor):
    errors = list()
    joints_3d_1 = joints_3d_1.copy()
    joints_3d_2 = joints_3d_2.copy()
    for i in [0, 21]:
        if joint_weights[i, 0] > 0: # hand wrist (right / left) is valid
            joints_3d_1 -= joints_3d_1[i:i+1, :]
            joints_3d_2 -= joints_3d_2[i:i+1, :]
            for j in range(21):
                if joint_weights[i+j, 0] > 0:
                    joints_1 = joints_3d_1[i+j]
                    joints_2 = joints_3d_2[i+j]
                    distance = np.linalg.norm( (joints_1-joints_2) )
                    distance /= scale_factor
                    errors.append(distance)
    return errors


def get_single_inter_joints_error(joints_3d_1, joints_3d_2, joint_weights, scale_factor):
    errors = list()
    joints_3d_1 = joints_3d_1.copy()
    joints_3d_2 = joints_3d_2.copy()
    i = 0
    if joint_weights[i, 0] > 0: # hand wrist (right / left) is valid
        joints_3d_1 -= joints_3d_1[i:i+1, :]
        joints_3d_2 -= joints_3d_2[i:i+1, :]
        for j in range(42):
            if joint_weights[i+j, 0] > 0:
                joints_1 = joints_3d_1[i+j]
                joints_2 = joints_3d_2[i+j]
                distance = np.linalg.norm( (joints_1-joints_2) )
                distance /= scale_factor
                errors.append(distance)
    return errors


def calc_transform(S1, S2):
    """
    Procrustes Analysis
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def calc_transform_no_rot(S1, S2):
    mean_1 = np.mean(S1, axis=0).reshape(1, 3)
    mean_2 = np.mean(S2, axis=0).reshape(1, 3)

    std_1 = np.std(S1, axis=0).reshape(1, 3)
    std_2 = np.std(S2, axis=0).reshape(1, 3)

    S1_norm = (S1 - mean_1) / std_1
    S1_trans = S1_norm * std_2 + mean_2

    return S1_trans


def get_single_pa_inter_joints_error(pred_joints, gt_joints, joints_valid, scale_factor, use_rot):
    assert len(pred_joints.shape) == 2
    assert len(gt_joints.shape) == 2
    if len(joints_valid.shape) == 2:
        assert joints_valid.shape[1] == 1
        joints_valid = joints_valid[:, 0]
    else:
        assert len(joints_valid.shape) == 1
    
    # print(joints_valid>0)
    # pdb.set_trace()
    if np.sum(joints_valid) < 2.0:
        return []

    pred_joints = pred_joints[joints_valid>0, :3]
    gt_joints = gt_joints[joints_valid>0, :3]


    transform_func = calc_transform if use_rot else calc_transform_no_rot
    pred_joints_trans = transform_func(pred_joints.copy(), gt_joints.copy())

    mpjpe_all = np.linalg.norm(pred_joints_trans-gt_joints, axis=1)
    mpjpe_all /= scale_factor
    return mpjpe_all.tolist()


def calc_collision_auc(collision_all):
    col_all = collision_all

    start = 0.5
    end = 15
    xs = list()
    ratios = list()
    for thresh in np.linspace(start, end):
        ratio = np.mean(col_all < thresh)
        x = (thresh-start) / (end-start)
        xs.append(x)
        ratios.append(ratio)

    auc = np.trapz(ratios, xs)
    return auc