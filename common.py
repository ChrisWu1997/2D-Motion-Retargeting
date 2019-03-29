import os
from functional import utils
import torch
import numpy as np


class Config:
    name = None
    device = None

    # data paths
    data_dir = '/data1/wurundi/mixamo_release'
    meanpose_path = None
    stdpose_path = None

    # training paths
    save_dir = './train_log'
    exp_dir = None
    log_dir = None
    model_dir = None

    # data info
    img_size = (512, 512)
    unit = 128
    nr_joints = 15
    len_joints = 2 * nr_joints - 2
    view_angles = [(0, 0, -np.pi / 2),
                   (0, 0, -np.pi / 3),
                   (0, 0, -np.pi / 6),
                   (0, 0, 0),
                   (0, 0, np.pi / 6),
                   (0, 0, np.pi / 3),
                   (0, 0, np.pi / 2)]

    # network channels
    mot_en_channels = None
    body_en_channels = None
    view_en_channels = None
    de_channels = None

    # training settings
    use_triplet = True
    triplet_margin = 1
    triplet_weight = 1

    nr_epochs = 300
    batch_size = 64
    num_workers = 4
    lr = 1e-3

    save_frequency = 50
    val_frequency = 100
    visualize_frequency = 500

    def initialize(self, args):
        self.name = args.name
        self.use_triplet = not args.disable_triplet if hasattr(args, 'disable_triplet') else None

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.exp_dir = os.path.join(self.save_dir, 'exp_' + self.name)
        self.log_dir = os.path.join(self.exp_dir, 'log/')
        self.model_dir = os.path.join(self.exp_dir, 'model/')
        utils.ensure_dirs([self.log_dir, self.model_dir])

        if self.name == 'skeleton':
            self.mot_en_channels = [self.len_joints + 2, 64, 96, 128]
            self.body_en_channels = [self.len_joints, 32, 48, 64]
            self.de_channels = [self.mot_en_channels[-1] + self.body_en_channels[-1], 128, 64, self.len_joints + 2]
            self.view_angles = None

            self.meanpose_path = os.path.join(self.data_dir, "meanpose.npy")
            self.stdpose_path = os.path.join(self.data_dir, "stdpose.npy")
        elif self.name == 'view':
            self.mot_en_channels = [self.len_joints + 2, 64, 96, 128]
            self.view_en_channels = [self.len_joints, 64, 96, 128, 32]
            self.de_channels = [self.mot_en_channels[-1] + self.view_en_channels[-1], 128, 64, self.len_joints + 2]

            self.meanpose_path = os.path.join(self.data_dir, "meanpose_with_view.npy")
            self.stdpose_path = os.path.join(self.data_dir, "stdpose_with_view.npy")
        else:
            self.mot_en_channels = [self.len_joints + 2, 64, 96, 128]
            self.body_en_channels = [self.len_joints, 32, 48, 64, 16]
            self.view_en_channels = [self.len_joints, 32, 48, 64, 8]
            self.de_channels = [self.mot_en_channels[-1] + self.body_en_channels[-1] + self.view_en_channels[-1],
                                128, 64, self.len_joints + 2]

            self.meanpose_path = os.path.join(self.data_dir, "meanpose_with_view.npy")
            self.stdpose_path = os.path.join(self.data_dir, "stdpose_with_view.npy")


config = Config()
