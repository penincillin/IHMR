
import os
import os.path as osp
import shutil
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


class BaseModel():

    @property
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor
        self.save_dir = osp.join(opt.checkpoints_dir)

    # helper saving function that can be used by subclasses
    def save_network(self, network, model_name, epoch, stage_id=None):
        if stage_id is None:
            save_filename = f"{epoch}_net_{model_name}"
        else:
            save_filename = f"{epoch}_net_{model_name}_stage_{stage_id:02d}"
        save_path = osp.join(self.save_dir, save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def save_info(self, save_info, epoch, stage_id=None):
        if stage_id is None:
            save_filename = f'{epoch}_info.pth'
        else:
            save_filename = f'{epoch}_info_stage_{stage_id:02d}.pth'
        save_path = osp.join(self.save_dir, save_filename)
        torch.save(save_info, save_path)

    # helper loading function that can be used by subclasses
    def load_network(self, network, model_name, epoch, stage_id=None):
        if stage_id is None:
            save_filename = f"{epoch}_net_{model_name}.pth"
        else:
            save_filename = f"{epoch}_net_{model_name}_stage_{stage_id:02d}.pth"
        save_path = osp.join(self.save_dir, save_filename)
        if not osp.exists(save_path):
            print(f"{save_path} does not exist !!!")
            return False
        else:
            if self.opt.dist:
                network.module.load_state_dict(torch.load(
                    save_path, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())))
            else:
                saved_weights = torch.load(save_path)
                network.load_state_dict(saved_weights)
            return True

    def load_info(self, epoch):
        save_filename = '{}_info.pth'.format(epoch)
        save_path = osp.join(self.save_dir, save_filename)
        # saved_info = torch.load(save_path)
        if self.opt.dist:
            saved_info = torch.load(save_path, map_location=lambda storage, loc: storage.cuda(
                torch.cuda.current_device()))
        else:
            saved_info = torch.load(save_path)
        return saved_info