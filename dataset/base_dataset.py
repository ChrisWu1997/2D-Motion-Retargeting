from functional.motion import trans_motion3d, normalize_motion, get_local3d
import os
from torch.utils.data import Dataset
import torch
import glob
import numpy as np


class _MixamoDatasetBase(Dataset):
    def __init__(self, phase, config):
        super(_MixamoDatasetBase, self).__init__()

        assert phase in ['train', 'test']
        self.data_root = os.path.join(config.data_dir, phase)
        self.phase = phase
        self.unit = config.unit
        self.view_angles = config.view_angles
        self.meanpose_path = config.meanpose_path
        self.stdpose_path = config.stdpose_path

        # FIXME : decide character_names
        if phase == 'train':
            self.character_names = ['Aj', 'BigVegas', 'Claire', 'Jasper', 'Lola', 'Malcolm',
                                    'Pearl', 'Warrok', 'Globin', 'Kaya', 'PeanutMan']
            self.aug = True
        else:
            self.character_names = ['Ty', 'Andromeda', 'Pumpkinhulk', 'SportyGranny']
            self.aug = False

        items = glob.glob(os.path.join(self.data_root, self.character_names[0], '*/motions/*.npy'))
        self.motion_names = ['/'.join(x.split('/')[-3:]) for x in items]

        self.mean_pose, self.std_pose = get_meanpose(config)

    def build_item(self, mot_name, char_name):
        """
        :param mot_name: animation_name/motions/xxx.npy
        :param char_name: character_name
        :return:
        """
        return os.path.join(self.data_root, char_name, mot_name)

    @staticmethod
    def gen_aug_param(rotate=False):
        if rotate:
            return {'ratio': np.random.uniform(0.8, 1.2),
                    'roll': np.random.uniform((-np.pi / 9, -np.pi / 9, -np.pi / 6), (np.pi / 9, np.pi / 9, np.pi / 6))}
        else:
            return {'ratio': np.random.uniform(0.5, 1.5)}

    @staticmethod
    def augmentation(data, param=None):
        """
        :param data: numpy array of size (joints, 3, len_frames)
        :return:
        """
        if param is None:
            return data, param

        # rotate
        if 'roll' in param.keys():
            cx, cy, cz = np.cos(param['roll'])
            sx, sy, sz = np.sin(param['roll'])
            mat33_x = np.array([
                [1, 0, 0],
                [0, cx, -sx],
                [0, sx, cx]
            ], dtype='float')
            mat33_y = np.array([
                [cy, 0, sy],
                [0, 1, 0],
                [-sy, 0, cy]
            ], dtype='float')
            mat33_z = np.array([
                [cz, -sz, 0],
                [sz, cz, 0],
                [0, 0, 1]
            ], dtype='float')
            data = mat33_x @ mat33_y @ mat33_z @ data

        # scale
        if 'ratio' in param.keys():
            data = data * param['ratio']

        return data, param

    def preprocessing(self, item, view_angle=None, param=None):
        """
        :param item: filename built from self.build_tiem
        :return:
        """
        motion3d = np.load(item)

        if self.aug:
            motion3d, param = self.augmentation(motion3d, param)

        # convert 3d to 2d
        local3d = None
        if view_angle is not None:
            local3d = get_local3d(motion3d, view_angle)

        motion_proj = trans_motion3d(motion3d, local3d, self.unit)
        motion_proj = normalize_motion(motion_proj, self.mean_pose, self.std_pose)
        motion_proj = motion_proj.reshape((-1, motion_proj.shape[-1]))   # reshape to (joints*2, len_frames)
        motion_proj = torch.Tensor(motion_proj) # FIXME : change to torch.from_numpy?
        return motion_proj

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.motion_names) * len(self.character_names)


def get_meanpose(config):
    meanpose_path = config.meanpose_path
    stdpose_path = config.stdpose_path
    if os.path.exists(meanpose_path) and os.path.exists(stdpose_path):
        meanpose = np.load(meanpose_path)
        stdpose = np.load(stdpose_path)
    else:
        meanpose, stdpose = gen_meanpose(config)
        np.save(meanpose_path, meanpose)
        np.save(stdpose_path, stdpose)
        print("meanpose saved at {}".format(meanpose_path))
        print("stdpose saved at {}".format(stdpose_path))
    return meanpose, stdpose


def gen_meanpose(config):
    all_paths = glob.glob(os.path.join(config.data_dir, 'train', '*/*/motions/*.npy'))
    all_joints = []

    for path in all_paths:
        motion3d = np.load(path)
        local3d = None
        if config.view_angles is None:
            motion_proj = trans_motion3d(motion3d, local3d)
            all_joints.append(motion_proj)
        else:
            for angle in config.view_angles:
                local3d = get_local3d(motion3d, angle)
                motion_proj = trans_motion3d(motion3d.copy(), local3d)
                all_joints.append(motion_proj)

    all_joints = np.concatenate(all_joints, axis=2)

    meanpose = np.mean(all_joints, axis=2)
    stdpose = np.std(all_joints, axis=2)
    stdpose[np.where(stdpose == 0)] = 1e-9
    return meanpose, stdpose
