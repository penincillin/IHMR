import argparse
import os
import os.path as osp
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dist', action='store_true', help='whether to use distributed training')
        self.parser.add_argument('--local_rank', type=int, default=0)
        self.parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
        self.parser.add_argument('--inputSize', type=int, default=224, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='h3dw', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=512,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=80, help='visdom port of the web display')

        self.parser.add_argument('--data_root', type=str, default='', help='root dir for all the datasets')
        self.parser.add_argument('--model_root', type=str, default='', help='root dir for all the pretrained weights and pre-defined models')
        self.parser.add_argument('--param_root', type=str, default='', help='root dir for all the pretrained weights and pre-defined models')
        self.parser.add_argument('--hand26m_anno_path', type=str, default='', help='path to file that stores the information of InterHand2.6M dataset')
        self.parser.add_argument('--hand26m_pred_path', type=str, default='', help='path to file that stores the predictions (joints, mano, image features) of InterHand.26M')

        self.parser.add_argument("--model_type", type=str, default='baseline', choices=['baseline', 'mlp', 'opt'])
        self.parser.add_argument('--num_joints', type=int, default=42, help='number of keypoints')
        self.parser.add_argument('--total_params_dim', type=int, default=122, help='number of params to be estimated') # to be determined
        self.parser.add_argument('--cam_params_dim', type=int, default=3, help='number of camera params to be estimated') # to be determined
        self.parser.add_argument('--pose_params_dim', type=int, default=48*2, help='number of hand pose params to be estimated') # to be determined
        self.parser.add_argument('--shape_params_dim', type=int, default=10*2, help='number of hand betas params to be estimated') # to be determined
        self.parser.add_argument('--trans_params_dim', type=int, default=3, help='dim of translation from right hand to left.')
        self.parser.add_argument('--mean_param_file', type=str, default='mean_mano_params.pkl', help='path of mano mean parameters')
        self.parser.add_argument('--main_encoder', type=str, default='resnet50', help='selects model to use for major input, it is usually image')
        self.parser.add_argument('--strategy', type=str, default='default', help='')

        #self.parser.add_argument("--joint_regressor_type", type=str, default="default")

        self.initialized = True


    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        return self.opt
