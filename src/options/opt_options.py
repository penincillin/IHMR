from .base_options import BaseOptions

class OptOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--joints_2d_loss_weight', type=float, default=10.0, help='')
        self.parser.add_argument('--joints_3d_loss_weight', type=float, default=10.0, help='')
        self.parser.add_argument('--pose_param_weight', type=float, default=10.0, help='')
        self.parser.add_argument('--shape_param_weight', type=float, default=10.0, help='')
        self.parser.add_argument('--trans_loss_weight', type=float, default=10.0, help='')
        self.parser.add_argument('--collision_loss_weight', type=float, default=1.0, help='')
        self.parser.add_argument('--shape_reg_loss_weight', type=float, default=0.1, help='')
        self.parser.add_argument('--shape_reg_loss_format', type=str, default='l2', choices=['l1','l2'], help='')
        self.parser.add_argument('--sdf_robustifier', type=float, default=None,)
        self.parser.add_argument('--use_hand_rotation', action='store_true', help='if specified, use ground truth hand rotation in training')
        self.parser.add_argument('--opt_dataset', type=str, help="which datasets to use in training")
        self.parser.add_argument('--save_mid_freq', type=int, default=1, help='')
        self.parser.add_argument("--optimizer", type=str, default='adam', choices=['adam', 'sgd'], help='')
        self.isTrain = False
