import os
import sys
sys.path.append('src/')
import shutil
import os.path as osp
import numpy as np
import pickle
import copy
import cv2
import time
import ry_utils
import ry_utils as ry_utils
import utils.metric_utils as mu
import utils.render_color_utils as rcu
import utils.vis_util as vu
import multiprocessing as mp
import pdb


class Evaluator(object):

    # def __init__(self, dataset_name, data_list, model):
    def __init__(self, opt, test_dataset, model):
        self.dataset_name = test_dataset.name 
        self.data_list = test_dataset.data_list
        self.image_root = test_dataset.image_root
        self.inputSize = model.inputSize
        self.left_hand_faces = model.mano_models['left'].faces
        self.right_hand_faces = model.mano_models['right'].faces
        self.pred_results = list()
    
    def gather_pred(self, pred_results):
        self.pred_results += pred_results

    def clear(self):
        self.pred_results = list()

    def update(self, data_idxs, pred_results, save_verts=True):
        self.save_verts = save_verts

        for i, data_idx in enumerate(data_idxs):
            single_data = dict(
                data_idx = data_idx,
                pred_cam_params = pred_results['pred_cam_params'][i],
                pred_shape_params = pred_results['pred_shape_params'][i],
                pred_pose_params = pred_results['pred_pose_params'][i],
                pred_hand_trans = pred_results['pred_hand_trans'][i],
                pred_joints_3d = pred_results['pred_joints_3d'][i],
                collision_loss_origin_scale = pred_results['collision_loss_origin_scale'][i],
                gt_joints_3d = pred_results['gt_joints_3d'][i],
                img_path = osp.join(self.image_root, self.data_list[data_idx]['img_path']),
                img_path_relative = self.data_list[data_idx]['img_path'],
            )
            default_values = dict(
                annot_type = 'machine',
                hand_type = 'interacting',
                hand_type_valid = 1.0,
                scale = 1.0,
            )
            for key in default_values:
                if key in self.data_list[data_idx]:
                    single_data[key] = self.data_list[data_idx][key]
                else:
                    single_data[key] = default_values[key]

            # save vertices for visualization
            if save_verts:
                for mode in ['pred', 'gt']:
                    for hand_type in ['left', 'right']:
                        key = f'{mode}_{hand_type}_hand_verts'
                        if key in pred_results:
                            single_data[key] = pred_results[key][i].astype(np.float16)

            # mpjpe for 3d joints
            pred_joints_3d = single_data['pred_joints_3d']
            gt_joints_3d = single_data['gt_joints_3d'][:, :3]
            joints_valid = single_data['gt_joints_3d'][:, 3:]
            scale_factor = single_data['scale']

            # mpjpe (separate left / right wrist)
            single_data['j3d_error'] = mu.get_single_joints_error(
                pred_joints_3d, gt_joints_3d, joints_valid, scale_factor)

            # pa-mpjpe (right wrist, without rotation)
            single_data['pa_no_rot_inter_j3d_error'] = mu.get_single_pa_inter_joints_error(
                pred_joints_3d, gt_joints_3d, joints_valid, scale_factor, use_rot=False)
            
            # acc of hand type
            hand_type = single_data['hand_type']
            hand_type_valid = single_data['hand_type_valid']
            
            # flip left-hand back
            if pred_results['do_flip'][i]:
                self.__flip_back_data(single_data)

            # update pred_results
            self.pred_results.append(single_data)


    def __flip_back_data(self, single_data):
        # cam
        single_data['pred_cam_params'][1] *= -1
        # trans
        single_data['pred_hand_trans'][0] *= -1
        # pose params
        pose_params = single_data['pred_pose_params'].copy()
        single_data['pred_pose_params'][:48] = pose_params[48:]
        single_data['pred_pose_params'][48:] = pose_params[:48]
        single_data['pred_pose_params'][1::3] *= -1
        single_data['pred_pose_params'][2::3] *= -1
        # joints
        for key in ['pred_joints_3d', 'gt_joints_3d']:
            joints_3d = single_data[key].copy()
            single_data[key][:21] = joints_3d[21:]
            single_data[key][21:] = joints_3d[:21]
            single_data[key][:, 0] *= -1
        # collision
        collision = single_data['collision_loss_origin_scale'].copy()
        single_data['collision_loss_origin_scale'][:778] = collision[778:]
        single_data['collision_loss_origin_scale'][778:] = collision[:778]
        # hand verts
        if self.save_verts:
            saved_data = dict()
            for mode in ['pred', 'gt']:
                for hand_type in ['left', 'right']:
                    key = f'{mode}_{hand_type}_hand_verts'
                    saved_data[key] = single_data[key].copy()
            for mode in ['pred', 'gt']:
                for hand_type in ['left', 'right']:
                    flip_hand_type = 'left' if hand_type == 'right' else 'right'
                    key = f'{mode}_{hand_type}_hand_verts'
                    key_flip = f'{mode}_{flip_hand_type}_hand_verts'
                    single_data[key] = saved_data[key_flip]
                    single_data[key][:, 0] *= -1


    def remove_redunc(self):
        new_pred_results = list()
        img_id_set = set()
        for data in self.pred_results:
            img_id = data['img_path_relative']
            if img_id not in img_id_set:
                new_pred_results.append(data)
                img_id_set.add(img_id)
        self.pred_results = new_pred_results
        print("Number of test data:", len(self.pred_results))
    

    @property
    def mpjpe_3d(self):
        all_errors = list()
        for pred in self.pred_results:
            all_errors += pred['j3d_error']
        return np.average(all_errors)
    
    @property
    def inter_mpjpe_3d(self):
        all_errors = list()
        for pred in self.pred_results:
            all_errors += pred['pa_no_rot_inter_j3d_error']
        return np.average(all_errors)
    
    @property
    def collision_ave(self):
        coll_all = list()
        for pred in self.pred_results:
            if pred['hand_type'] == 'interacting':
                coll_os = pred['collision_loss_origin_scale']
                coll_ave = np.mean(coll_os) * 1000
                coll_all.append(coll_ave)
        return np.average(coll_all)
    
    @property
    def collision_max(self):
        coll_all = list()
        for pred in self.pred_results:
            if pred['hand_type'] == 'interacting':
                coll_os = pred['collision_loss_origin_scale']
                coll_max = np.max(coll_os) * 1000
                coll_all.append(coll_max)
        return np.average(coll_all)


    def __build_dirs(self, res_dir):
        for pred in self.pred_results:
            record = pred['img_path'].split('/')
            img_name = osp.join(record[-4], record[-3], '_'.join(record[-2:]))
            pred['img_name'] = img_name
            res_img_path = osp.join(res_dir, img_name)
            ry_utils.make_subdir(res_img_path)
    
    def __pad_and_resize(self, img, final_size=224):
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
        return new_img

    def __render_two_hand(self, data, img):
        verts_list = [data['pred_right_hand_verts'], data['pred_left_hand_verts']]
        faces_list = [self.right_hand_faces, self.left_hand_faces]

        verts0, verts1 = verts_list[0], verts_list[1]
        faces0, faces1 = faces_list[0], faces_list[1]
        verts = np.concatenate((verts0, verts1), axis=0)
        faces = np.concatenate((faces0, faces1+verts0.shape[0]), axis=0)

        color0 = np.array(rcu.colors['light_green']).reshape(1, 3)
        color1 = np.array(rcu.colors['light_blue']).reshape(1, 3)
        color_list = [color0, color1]
        cam = data['pred_cam_params']
        render_img = rcu.render_together(
            verts_list, faces_list, color_list, cam, img.shape[0], img)
        return render_img, verts, faces

    def __render_single_hand(self, data, img):
        hand_type = data['hand_type']
        verts = data[f'pred_{hand_type}_hand_verts']
        faces = getattr(self, f'{hand_type}_hand_faces')
        cam = data['pred_cam_params']
        render_img = vu.render(verts, faces, cam, img.shape[0], img)
        return render_img, verts, faces

    def __visualize_result(self, start, end, res_vis_dir, res_obj_dir, size_type='double'):
        for i, result in enumerate(self.pred_results[start:end]):
            img_path = result['img_path']
            ori_img = cv2.imread(img_path)

            # resize img
            if size_type == 'origin':
                final_size = np.max(ori_img.shape[:2])
            elif size_type == 'double':
                final_size = self.inputSize * 2 # 224 * 2
            else:
                assert size_type == 'normalized' # 224
                final_size = self.inputSize
            img = self.__pad_and_resize(ori_img, final_size)

            if result['hand_type'] == 'interacting':
                render_img, verts, faces = self.__render_two_hand(result, img)
            else:
                render_img, verts, faces = self.__render_single_hand(result, img)
            res_img = np.concatenate((img, render_img))

            # save results
            img_name = result['img_name']
            res_img_path = osp.join(res_vis_dir, img_name).replace(".png", ".jpg")
            cv2.imwrite(res_img_path, res_img)
            res_obj_path = osp.join(res_obj_dir, img_name)[:-4] + '.obj'
            ry_utils.save_mesh_to_obj(res_obj_path, verts, faces)
            if i%10 == 0:
                print("{} Processed:{}/{}".format(os.getpid(), i, end-start))
    

    def visualize_result(self, res_vis_dir, res_obj_dir):
        num_process = min(len(self.pred_results), 16)
        num_each = len(self.pred_results) // num_process
        process_list = list()
        self.__build_dirs(res_vis_dir)
        self.__build_dirs(res_obj_dir)
        for i in range(num_process):
            start = i*num_each
            end = (i+1)*num_each if i<num_process-1 else len(self.pred_results)
            p = mp.Process(target=self.__visualize_result, args=(start, end, res_vis_dir, res_obj_dir))
            p.start()
            process_list.append(p)
        for p in process_list:
            p.join()


def main():
    method = sys.argv[1]
    dataset = sys.argv[2]

    pkl_path = osp.join('evaluate_results', method, f'{dataset}.pkl')
    assert osp.exists(pkl_path)
    evaluator = ry_utils.load_pkl(pkl_path)

    res_vis_dir = osp.join('evaluate_results', method, dataset, 'images')
    res_obj_dir = osp.join('evaluate_results', method, dataset, 'objs')
    ry_utils.renew_dir(res_vis_dir)
    ry_utils.renew_dir(res_obj_dir)

    evaluator.visualize_result(res_vis_dir, res_obj_dir)


if __name__ == '__main__':
    main()
