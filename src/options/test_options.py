from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--visualize_eval', action='store_true')
        self.parser.add_argument('--test_dataset', type=str, help="which dataset to test on")
        self.parser.add_argument('--test_epoch', type=str, help="Use epoch to test the model")
        self.isTrain = False
