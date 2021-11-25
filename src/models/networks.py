import torch
import torch.nn as nn
from torch.nn import init
import functools
import numpy as np
from . import resnet

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def get_model(arch):
    if hasattr(resnet, arch):
        network = getattr(resnet, arch)
        return network(pretrained=True, num_classes=512)
    else:
        raise ValueError("Invalid Backbone Architecture")


class InterHandEncoder(nn.Module):
    def __init__(self, opt, mean_params):
        super(InterHandEncoder, self).__init__()
        self.mean_params = mean_params.clone().cuda()
        self.opt = opt
        self.main_encoder = get_model(opt.main_encoder)

        relu = nn.ReLU(inplace=False)
        fc2  = nn.Linear(1024, 1024)
        feat_encoder = [relu, fc2, relu]
        self.feat_encoder = nn.Sequential(*feat_encoder)

        regressor = nn.Linear(1024 + opt.total_params_dim, opt.total_params_dim)
        regressor = [regressor, ]
        self.regressor_ih = nn.Sequential(*regressor) # ih stands for inter hand

        hand_classifier = nn.Linear(1024, 2)
        hand_classifier = [hand_classifier, ]
        self.hand_classifier = nn.Sequential(*hand_classifier)


    def forward(self, main_input):
        main_feat = self.main_encoder(main_input)
        feat = self.feat_encoder(main_feat)

        # parameters
        pred_params = self.mean_params
        for i in range(3):
            input_feat = torch.cat([feat, pred_params], dim=1)
            output = self.regressor_ih(input_feat)
            pred_params = pred_params + output
        
        # hand classifier
        hand_class = torch.sigmoid(self.hand_classifier(feat))

        return pred_params, hand_class


class InterHandSubNetwork(nn.Module):
    def __init__(self, opt, input_dim, update_param_dim):
        super(InterHandSubNetwork, self).__init__()
        self.opt = opt
        relu = nn.ReLU(inplace=True)

        # in_dim = 1024 + opt.total_params_dim
        fc1 = nn.Linear(input_dim, 512)
        fc2 = nn.Linear(512, 256)
        fc3 = nn.Linear(256, 128)
        regressor = nn.Linear(128, update_param_dim)

        nn.init.xavier_uniform_(fc1.weight, gain=0.01)
        nn.init.xavier_uniform_(fc2.weight, gain=0.01)
        nn.init.xavier_uniform_(fc3.weight, gain=0.01)
        nn.init.xavier_uniform_(regressor.weight, gain=0.01)

        layers = [fc1, relu, fc2, relu, fc3, relu, regressor]
        self.regressor = nn.Sequential(*layers)
    
    def forward(self, inputs):
        output = self.regressor(inputs)
        return output
