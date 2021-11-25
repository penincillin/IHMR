from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=2048, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=2048, help='frequency of showing training results on console')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--total_epoch', type=int, default=100, help='the number of epoch we need to train the model')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load?')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--lr', type=float, default=1e-5, help='initial learning rate for encoder') 
        self.parser.add_argument('--lr_decay_type', type=str, choices=['none', 'stage', 'cosine'], default='none', help='')
        self.parser.add_argument('--joints_2d_loss_weight', type=float, default=10.0, help='')
        self.parser.add_argument('--joints_3d_loss_weight', type=float, default=10.0, help='')
        self.parser.add_argument('--pose_param_weight', type=float, default=10.0, help='')
        self.parser.add_argument('--shape_param_weight', type=float, default=10.0, help='')
        self.parser.add_argument('--trans_loss_weight', type=float, default=10.0, help='')
        self.parser.add_argument('--collision_loss_weight', type=float, default=1.0, help='')
        self.parser.add_argument('--shape_reg_loss_weight', type=float, default=0.1, help='')
        self.parser.add_argument('--shape_reg_loss_format', type=str, default='l2', choices=['l1','l2'], help='')
        self.parser.add_argument('--sdf_robustifier', type=float, default=None,)
        self.parser.add_argument('--use_collision_loss', action='store_true', default=None,)
        self.parser.add_argument('--use_hand_rotation', action='store_true', help='if specified, use ground truth hand rotation in training')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/web/')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--pretrain_weights', type=str, default=None, help='path to pretrained weights')
        self.parser.add_argument('--pretrain_weights_dir', type=str, default=None, help='path to pretrained weights')

        self.parser.add_argument('--train_datasets', type=str, help="which datasets to use in training")
        self.parser.add_argument("--use_random_rescale", action='store_true', help='use random rescale in data augmentation')
        self.parser.add_argument("--use_random_position", action='store_true', help='use random position in data augmentation')
        self.parser.add_argument("--use_random_flip", action='store_true', help='use random position in data augmentation')
        self.parser.add_argument("--use_random_rotation", action='store_true', help='use random rotation in data augmentation')
        self.parser.add_argument("--use_color_jittering", action="store_true", help="use color jittering in data augmentation")
        self.parser.add_argument("--use_motion_blur", action="store_true", help="use motion blur augmentation")
        self.parser.add_argument("--blur_kernel_dir", type=str, default="path of directory that stores blur kernel")
        self.parser.add_argument("--motion_blur_prob", type=float, default=0.5, help="the probability of using motion blur")
        self.parser.add_argument("--num_sample", type=int, default=10000,)
        self.parser.add_argument("--use_opt_params", action='store_true', help="Use parameters obatined from IHMR-OPT in training")
        self.isTrain = True