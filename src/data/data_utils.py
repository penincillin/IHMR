import os
import os.path as osp
import ry_utils
from scipy.io import loadmat

def _load_data_from_dir(img_dir):
    data_list = list()
    for subdir, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith( ("jpg", "jpeg", "png") ):
                img_name = osp.join(subdir, file).replace(img_dir, "")
                if img_name[0] == "/": # remove the potential "/" in the head of path
                    img_name = img_name[1:]
                single_data = dict(
                    image_name = img_name
                )
                data_list.append(single_data)
    assert len(data_list)>0, "Given Directory contains no image."
    return data_list


def load_annotation(data_root, anno_path, use_augment=True):
    anno_path_full = osp.join(data_root, anno_path)
    if osp.isdir(anno_path_full):
        all_data = _load_data_from_dir(anno_path_full)
    else:
        all_data = ry_utils.load_pkl(anno_path_full)
    
    if isinstance(all_data, list):
        data_list = all_data
    else:
        raise ValueError("Unsupported data type")
    return data_list


def __load_data_to_dict(all_data, key):
    res_data = dict()
    for data in all_data:
        res_data[data[key]] = data
    return res_data

def load_anno_pred_data(data_root, anno_path, pred_res_path):
    # annotation
    anno_path_full = osp.join(data_root, anno_path)
    anno_data_all = ry_utils.load_pkl(anno_path_full)
    anno_data_all = __load_data_to_dict(anno_data_all, key='img_path')

    # prediction
    pred_res_path_full = osp.join(data_root, pred_res_path)
    pred_res_all = ry_utils.load_pkl(pred_res_path_full)

    data_list = list()
    for key in anno_data_all:
        anno_data = anno_data_all[key]
        pred_res = pred_res_all[key]
        # print(pred_res.keys())
        # predicted motion
        for motion_key in ['pred_cam_params', 'pred_shape_params', 'pred_pose_params', 'pred_hand_trans']:
            anno_data[motion_key] = pred_res[motion_key]
        # predicted joints
        for joints_key in ['joints_2d', 'joints_3d']:
            anno_data[f'pred_{joints_key}'] = pred_res[joints_key]
        # image feature 
        for feat_key in ['img_feat']:
            anno_data['img_feat'] = pred_res[feat_key]

        data_list.append(anno_data)

    assert len(data_list)>0, f"Data List must have data."
    return data_list


def load_blur_kernel(blur_kernel_dir):
    kernels = list()
    for file in os.listdir(blur_kernel_dir):
        if file.endswith(".mat"):
            kernel = loadmat(osp.join(blur_kernel_dir, file))['PSFs'][0][0]
            kernels.append(kernel)
    return kernels