import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import glob
from copy import copy


class _MixamoDatasetBase(Dataset):
    def __init__(self, phase, data_dir):
        super(_MixamoDatasetBase, self).__init__()

        assert phase in ['train', 'test']
        self.data_root = os.path.join(data_dir, phase)
        self.phase = phase

        # FIXME : decide character_names
        if phase == 'train':
            self.character_names = ['Aj', 'BigVegas', 'Claire', 'Jasper', 'Lola', 'Malcolm',
                                    'Pearl', 'Warrok', 'Globin', 'Kaya', 'PeanutMan']
            self.aug = True
        else:
            self.character_names = ['Ty', 'Andromeda', 'Pumpkinhulk', 'SportyGranny', 'Whiteclown']
            self.aug = False

        items = glob.glob(os.path.join(self.data_root, self.character_names[0], '*/motions/*.npy'))
        self.motion_names = ['/'.join(x.split('/')[-3:]) for x in items]

    def build_item(self, char_name, mot_name):
        """
        :param char_name: character_name
        :param mot_name: animation_name/motions/xxx.npy
        :return:
        """
        return os.path.join(self.data_root, char_name, mot_name)

    @staticmethod
    def gen_aug_param(rotate=False):
        if rotate:
            return {'ratio': np.random.uniform(0.8, 1.2),
                    'roll': np.random.uniform((-np.pi / 9, -np.pi / 9, -np.pi / 6), (np.pi / 9, np.pi / 9, np.pi / 6))}
        else:
            return {'ratio': np.random.uniform(0.8, 1.2)}

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

    def __len__(self):
        return len(self.motion_names) * len(self.character_names)


class MixamoDatasetForSkeleton(_MixamoDatasetBase):
    def __init__(self, phase, data_dir):
        super(MixamoDatasetForSkeleton, self).__init__(phase, data_dir)

    def preprocessing(self, item, param=None):
        motion3d = np.load(item)

        if self.aug:
            motion3d, param = self.augmentation(motion3d, param)

        motion_proj = trans_motion(motion3d)

        motion_proj = (motion_proj - self.mean_pose[:, :, np.newaxis]) / self.std_pose[:, :, np.newaxis]

        motion_proj = motion_proj.reshape((-1, motion_proj.shape[-1]))   # reshape to (joints*2, len_frames)

        motion_proj = torch.Tensor(motion_proj)

        return motion_proj

    def __getitem__(self, index):
        # select two motions
        idx1, idx2 = np.random.choice(len(self.motion_names), size=2, replace=False)
        mot1, mot2 = self.motion_names[idx1], self.motion_names[idx2]
        # select two characters
        idx1, idx2 = np.random.choice(len(self.character_names), size=2, replace=False)
        char1, char2 = self.character_names[idx1], self.character_names[idx2]

        if self.aug:
            param1 = self.gen_aug_param(rotate=True)
            param2 = self.gen_aug_param(rotate=True)
            param12 = copy(param1)
            param21 = copy(param2)
            param12['ratio'] = param2['ratio']
            param21['ratio'] = param1['ratio']
        else:
            param1 = param2 = param12 = param21 = None

        item1 = self.build_item(mot1, char1)
        item2 = self.build_item(mot2, char2)
        item12 = self.build_item(mot1, char2)
        item21 = self.build_item(mot2, char1)

        input1 = self.preprocessing(item1, param1)
        input2 = self.preprocessing(item2, param2)
        target1 = input1.detach().clone()
        target2 = input2.detach().clone()

        input12 = self.preprocessing(item12, param12)
        input21 = self.preprocessing(item21, param21)
        target12 = input12.detach().clone()
        target21 = input21.detach().clone()

        return {"input1": input1, "target1": target1,
                "input2": input2, "target2": target2,
                "input12": input12, "target12": target12,
                "input21": input21, "target21": target21,
                "mot1": mot1, "mot2": mot2,
                "char1": char1, "char2": char2}


class MixamoDatasetForView(_MixamoDatasetBase):
    def __init__(self, phase, data_dir):
        super(MixamoDatasetForView, self).__init__(phase, data_dir)
        self.view_angles = [(0, 0, -np.pi / 2),
                           (0, 0, -np.pi / 3),
                           (0, 0, -np.pi / 6),
                           (0, 0, 0),
                           (0, 0, np.pi / 6),
                           (0, 0, np.pi / 3),
                           (0, 0, np.pi / 2)]

    def preprocessing(self, item, view_angle, param=None):
        """
        :param data: numpy array of size (joints, 3, len_frames)
        :return:
        """
        motion3d = np.load(item[0])[:15]
        local3d = np.load(item[1])

        # convert 3d to 2d
        local3d = rotate_coordinates(local3d, view_angle)

        motion_proj = trans_motion(motion3d, local3d, scale = param[0])

        motion_proj = (motion_proj - self.mean_pose[:, :, np.newaxis]) / self.std_pose[:, :, np.newaxis]
        motion_proj = motion_proj.reshape((-1, motion_proj.shape[-1]))   # reshape to (joints*2, len_frames)

        motion_proj = torch.Tensor(motion_proj)

        return motion_proj

    def __getitem__(self, index):
        # select two motions
        idx1, idx2 = np.random.choice(len(self.motion_names), size=2, replace=False)
        mot1, mot2 = self.motion_names[idx1], self.motion_names[idx2]
        # select two characters
        idx1, idx2 = np.random.choice(len(self.character_names), size=2)
        char1, char2 = self.character_names[idx1], self.character_names[idx2]
        # select two views
        idx1, idx2 = np.random.choice(len(self.view_angles), size=2, replace=False)
        view1, view2 = self.view_angles[idx1], self.view_angles[idx2]

        item1 = self.build_item(mot1, char1)
        item2 = self.build_item(mot2, char2)
        item12= self.build_item(mot1, char2)
        item21 = self.build_item(mot2, char1)

        if self.aug:
            param1 = self.gen_aug_param(rotate=False)   # FIXME: [np.random.uniform(0.5, 1.5)]
            param2 = self.gen_aug_param(rotate=False)
            param12 = param2
            param21 = param1
        else:
            param1 = param2 = param12 = param21 = None

        input1 = self.preprocessing(item1, view1, param1)
        input2 = self.preprocessing(item2, view2, param2)
        target1 = input1.detach().clone()
        target2 = input2.detach().clone()

        input122 = self.preprocessing(item12, view2, param12)
        input211 = self.preprocessing(item21, view1, param21)
        target122 = input122.detach().clone()
        target211 = input211.detach().clone()

        return {"input1": input1, "target111": target1,
                "input2": input2, "target222": target2,
                "input122": input122, "target122": target122,
                "input211": input211, "target211": target211,
                "mot1": mot1, "mot2": mot2,
                "view1": view1, "view2": view2,
                "char1": char1, "char2": char2}


class MixamoDatasetForThree(_MixamoDatasetBase):
    def __init__(self, phase, data_dir):
        super(MixamoDatasetForThree, self).__init__(phase, data_dir)
        self.view_angles = [(0, 0, -np.pi / 2),
                           (0, 0, -np.pi / 3),
                           (0, 0, -np.pi / 6),
                           (0, 0, 0),
                           (0, 0, np.pi / 6),
                           (0, 0, np.pi / 3),
                           (0, 0, np.pi / 2)]

    def preprocessing(self, item, view_angle, param=None):
        """
        :param data: numpy array of size (joints, 3, len_frames)
        :return:
        """
        motion3d = np.load(item[0])
        local3d = np.load(item[1])

        # convert 3d to 2d
        local3d = rotate_coordinates(local3d, view_angle)

        motion_proj = trans_motion(motion3d, local3d, scale = param[0])

        motion_proj = (motion_proj - self.mean_pose[:, :, np.newaxis]) / self.std_pose[:, :, np.newaxis]
        motion_proj = motion_proj.reshape((-1, motion_proj.shape[-1]))   # reshape to (joints*2, len_frames)

        motion_proj = torch.Tensor(motion_proj)

        return motion_proj

    def __getitem__(self, index):
        # select two motions
        idx1, idx2 = np.random.choice(len(self.motion_names), size=2, replace=False)
        mot1, mot2 = self.motion_names[idx1], self.motion_names[idx2]
        # select two characters
        idx1, idx2 = np.random.choice(len(self.character_names), size=2, replace=False)
        char1, char2 = self.character_names[idx1], self.character_names[idx2]
        # select two views
        idx1, idx2 = np.random.choice(len(self.view_angles), size=2, replace=False)
        view1, view2 = self.view_angles[idx1], self.view_angles[idx2]

        item1 = self.build_item(mot1, char1)
        item2 = self.build_item(mot2, char2)
        item12= self.build_item(mot1, char2)
        item21 = self.build_item(mot2, char1)

        if self.aug:
            param1 = self.gen_aug_param(rotate=False)   # FIXME: [np.random.uniform(0.5, 1.5)]
            param2 = self.gen_aug_param(rotate=False)
        else:
            param1 = param2 = None

        input1 = self.preprocessing(item1, view1, param1)
        input2 = self.preprocessing(item2, view2, param2)
        target1 = input1.detach().clone()
        target2 = input2.detach().clone()

        target112 = self.preprocessing(item1, view2, param1)
        target121 = self.preprocessing(item12, view1, param2)
        target122 = self.preprocessing(item12, view2, param2)
        target221 = self.preprocessing(item2, view1, param2)
        target212 = self.preprocessing(item21, view2, param1)
        target211 = self.preprocessing(item21, view1, param1)

        return {"input1": input1, "target111": target1,
                "input2": input2, "target222": target2,
                "target112": target112,
                "target121": target121,
                "target122": target122,
                "target221": target221,
                "target212": target212,
                "target211": target211,
                "mot1": mot1, "mot2": mot2,
                "view1": view1, "view2": view2,
                "char1": char1, "char2": char2}
