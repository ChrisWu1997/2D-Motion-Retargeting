# buggy! draw data samples with replacement.
import os
from torch.utils.data import Dataset, DataLoader
# Ignore warnings
import warnings
from common import config
import numpy as np
import torch
from motion import trans_motion, trans_motion2d, rotate_coordinates
warnings.filterwarnings("ignore")

DATA_ROOT = '/data1/wurundi/mixamo/mixamo-3d-data-face+hand+toe'
TRAIN_CHARACTERS = ['Aj', 'BigVegas', 'Claire', 'Jasper', 'Lola', 'Pearl', 'Warrok', 'Globin', 'Kaya']
VALIDATION_CHARACTERS = ['Ty', 'Pumpkinhulk', 'SportyGranny']
VIEW_RANGES = [(0, 0, -np.pi / 2),
               (0, 0, -np.pi / 3),
               (0, 0, -np.pi / 6),
               (0, 0, 0),
               (0, 0, np.pi / 6),
               (0, 0, np.pi / 3),
               (0, 0, np.pi / 2)]
MEAN_POSE_PATH = os.path.join('./model/meanpose_bodyview_7view-gtvLPP-noroot.npy')
STD_POSE_PATH = os.path.join('./model/stdpose_bodyview_7view-gtvLPP-noroot.npy')
FAILED_MOTIONS_PATH = os.path.join(DATA_ROOT, 'handtoe_7view_failed.npy')


class MixamoDataset(Dataset):
    def __init__(self, name):
        super(MixamoDataset, self).__init__()
        self.data_root = DATA_ROOT

        if name == 'train':
            self.character_names = TRAIN_CHARACTERS
            self.aug = True
        elif name == 'validation':
            self.character_names = VALIDATION_CHARACTERS
            self.aug = False
        else:
            raise NameError

        failed_motions = np.load(FAILED_MOTIONS_PATH)

        self.animation_names = sorted(os.listdir(os.path.join(self.data_root, self.character_names[0])))

        self.view_angles = VIEW_RANGES
        self.motion_names = []
        char0_dir = os.path.join(self.data_root, self.character_names[0])
        for anim in self.animation_names:
            mot_dir = os.path.join(char0_dir, anim, 'motions')
            nr_motions = len(os.listdir(mot_dir))
            for i in range(nr_motions):
                mot_name = os.path.join(anim, 'motions/{}.npy'.format(i + 1))
                if mot_name in failed_motions:
                    continue
                self.motion_names.append((mot_name, mot_name.replace('motions', 'locals')))

        self.mean_pose, self.std_pose = get_meanpose()

    def get_cluster_data(self, nr_motions=50):
        """
        get a certain subset data for clustering visualization and scoring
        :param nr_anims:
        :return: pre-processed data of shape (nr_views, nr_anims, 30, 64)
        """
        motion_items = self.motion_names
        if nr_motions < len(self.motion_names):
            idxes = np.linspace(0, len(self.motion_names) - 1, nr_motions, dtype=int)
            motion_items = [self.motion_names[i] for i in idxes]
        motion_names = np.array([item[0] for item in motion_items])

        all_data = []
        for mot in motion_items:
            char_data = []
            for char in self.character_names:
                item = self.build_item(mot, char)
                view_data = []
                for view in self.view_angles:
                    data = self.preprocessing(item, view, [1.0])
                    view_data.append(data)
                char_data.append(torch.stack(view_data, dim=0))
            all_data.append(torch.stack(char_data, dim=0))
        all_data = torch.stack(all_data, dim=0)

        ret = (all_data, motion_names, self.character_names, np.rad2deg(self.view_angles))
        return ret


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

    def build_item(self, mot, char):
        return (os.path.join(self.data_root, char, mot[0]),
                os.path.join(self.data_root, char, mot[1]))

    def __getitem__(self, index):
        # select two motions
        idx1, idx2 = np.random.choice(len(self.motion_names), size=2, replace=False)
        mot1, mot2 = self.motion_names[idx1], self.motion_names[idx2]
        # select two characters
        idx1, idx2 = np.random.choice(len(self.character_names), size=2)
        char1, char2 = self.character_names[idx1], self.character_names[idx2]
        # select two views
        idx1, idx2 = np.random.choice(len(self.view_angles), size=2)
        view1, view2 = self.view_angles[idx1], self.view_angles[idx2]

        item1 = self.build_item(mot1, char1)
        item2 = self.build_item(mot2, char2)
        item12= self.build_item(mot1, char2)
        item21 = self.build_item(mot2, char1)

        if self.aug:
            param1 = [np.random.uniform(0.5, 1.5)]
            param2 = [np.random.uniform(0.5, 1.5)]
        else:
            param1 = [1.0]
            param2 = [1.0]

        input1 = self.preprocessing(item1, view1, param1)
        input2 = self.preprocessing(item2, view2, param2)
        target1 = input1.detach().clone()
        target2 = input2.detach().clone()

        target122 = self.preprocessing(item12, view2, param2).detach()
        target211 = self.preprocessing(item21, view1, param1).detach()


        return {"input1": input1, "target111": target1,
                "input2": input2, "target222": target2,
                "target122": target122,
                "target211": target211,
                "mot1": mot1, "mot2": mot2,
                "view1": view1, "view2": view2,
                "char1": char1, "char2": char2}

    def __len__(self):
        return len(self.motion_names) * len(self.character_names)


def get_dataloaders(name='train', batch_size=8, shuffle=True, num_workers=4):
    """
    :param name: 'train' or 'validation' or 'test'
    :param batch_size: the size of an batch
    :param shuffle: whether random shuffle the data or not
    :param num_workers: the number of workers
    :return: the dataloader
    """
    dataset = MixamoDataset(name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, worker_init_fn=lambda _: np.random.seed())

    return dataloader


def gen_meanpose(data_root, character_names):
    animation_names = os.listdir(os.path.join(data_root, character_names[0]))

    all_joints = []
    failed_anim = np.load(FAILED_MOTIONS_PATH)
    for char in character_names:
        char_dir = os.path.join(data_root, char)
        for anim in animation_names:
            mot_dir = os.path.join(char_dir, anim, 'motions')
            local_dir = os.path.join(char_dir, anim, 'locals')
            nr_motions = len(os.listdir(mot_dir))
            for i in range(nr_motions):
                motion3d = np.load(os.path.join(mot_dir, '{}.npy'.format(i + 1)))
                local3d = np.load(os.path.join(local_dir, '{}.npy'.format(i + 1)))

                for view in VIEW_RANGES:
                    mot_name = os.path.join(anim, 'motions/{}.npy'.format(i + 1))
                    if mot_name in failed_anim:
                        continue
                    local3d = rotate_coordinates(local3d, view)
                    motion_proj = trans_motion(motion3d, local3d)
                    if motion_proj is None:
                        failed_anim.append(os.path.join(anim, 'motions', '{}.npy'.format(i + 1)))
                        continue
                    all_joints.append(motion_proj)

    all_joints = np.concatenate(all_joints, axis=2)
    meanpose = np.mean(all_joints, axis=2)
    stdpose = np.std(all_joints, axis=2)
    stdpose[np.where(stdpose == 0)] = 1e-9
    print(meanpose)
    print(stdpose)
    failed_anim = np.unique(failed_anim)
    print(len(failed_anim))
    np.save(MEAN_POSE_PATH, meanpose)
    np.save(STD_POSE_PATH, stdpose)
    #np.save(FAILED_MOTIONS_PATH, failed_anim)


def get_meanpose():
    if not os.path.exists(MEAN_POSE_PATH):
        gen_meanpose(DATA_ROOT, TRAIN_CHARACTERS)

    mean_pose = np.load(MEAN_POSE_PATH)
    std_pose = np.load(STD_POSE_PATH)
    return mean_pose, std_pose


MEAN_POSE, STD_POSE = get_meanpose()


def test():
    train_ds = get_dataloaders('train', batch_size=1)
    print(len(train_ds))
    for i, data in enumerate(train_ds):
        print(data['mot1'][0])
        print(data['mot2'][0])
        print(data['char1'][0])
        print(data['char2'][0])
        print(data['view1'])
        print(data['view2'])
        print(data['input1'][0].shape)
        break


if __name__=='__main__':
    #gen_meanpose(DATA_ROOT, VALIDATION_CHARACTERS)
    meanpose, std_pose = get_meanpose()
    print(meanpose.shape, std_pose.shape)
    print(meanpose)
    print(std_pose)
    test()
