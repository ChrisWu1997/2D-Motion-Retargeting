import sys
sys.path.append('../../functional')
import os
import json
import utils


class Config:
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #proj_dir = '/data1/wurundi/motion_disentangle/'
    proj_dir = './'
    exp_name = os.getcwd().split('/')[-1]
    exp_dir = os.path.join(proj_dir, exp_name)
    log_dir = os.path.join(exp_dir, 'log/')
    model_dir = os.path.join(exp_dir, 'model/')
    stat_path = os.path.join(proj_dir, 'statistic-body-face.csv')

    img_size = (512, 512)
    unit = 128
    len_frames = 64
    len_joints = 96

    foot_idx = [24, 25, 32, 33]
    fv_weight = 0.5

    mot_en_channels = [len_joints + 2, 128, 192, 256]
    body_en_channels = [len_joints, 128, 192, 256, 64]
    de_channels = [mot_en_channels[-1] + body_en_channels[-1], 192, 128, len_joints + 2]

    nr_epochs = 300
    batch_size = 64
    lr = 1e-3

    save_frequency = 50
    val_frequency = 100
    visualize_frequency = 8000

    utils.ensure_dirs([proj_dir, log_dir, exp_dir, model_dir])


config = Config()
