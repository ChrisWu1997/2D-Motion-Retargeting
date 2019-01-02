import os
from torch.utils.data import Dataset, DataLoader
# Ignore warnings
from common import config
import numpy as np
import torch
from copy import copy
from motion import trans_motion, normalize_motion

DATA_ROOT = '/data1/wurundi/mixamo/mixamo-3d-data-face+hand+toe'
TRAIN_CHARACTERS = ['Aj', 'BigVegas', 'Claire', 'Jasper', 'Lola', 'Malcolm', 'Pearl', 'Warrok', 'Globin', 'Kaya']
VALIDATION_CHARACTERS = ['Ty', 'Andromeda', 'Pumpkinhulk', 'SportyGranny']
MEAN_POSE_PATH = os.path.join('./model/meanpose_11char_gtdGOP-noroot.npy')
STD_POSE_PATH = os.path.join('./model/stdpose_11char_gtdGOP-noroot.npy')


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

        self.animation_names = sorted(os.listdir(os.path.join(self.data_root, self.character_names[0])))

        self.items = []
        for char in self.character_names:
            for anim in self.animation_names:
                mot_dir = os.path.join(self.data_root, char, anim, 'motions')
                nr_motions = len(os.listdir(mot_dir))
                self.items.extend([os.path.join(mot_dir, '{}.npy'.format(i + 1))
                                   for i in range(nr_motions)])

        self.mean_pose, self.std_pose = get_meanpose()

    @staticmethod
    def parse_item(item):
        """
        :param item: "data_root/character_name/animation_name/motions/xxx.npy"
        :return:
        """
        parts = item.split('/')

        return parts[-4], parts[-3] + '/motions/' + parts[-1]

    def build_item(self, char_name, mot_name):
        """
        :param char_name: character_name
        :param mot_name: animation_name/motions/xxx.npy
        :return:
        """
        return "{}/{}/{}".format(self.data_root, char_name, mot_name)

    def cross_2items(self, item1, item2):
        char1, mot1 = self.parse_item(item1)
        char2, mot2 = self.parse_item(item2)
        item12 = self.build_item(char2, mot1)
        item21 = self.build_item(char1, mot2)

        return item12, item21

    def get_cluster_data(self, nr_anims=100):
        """
        get a certain subset data for clustering visualization and scoring
        :param nr_anims:
        :return: pre-processed data of shape (nr_chars, nr_anims, 32, 64)
        """
        if nr_anims < len(self.animation_names):
            idxes = np.linspace(0, len(self.animation_names) - 1, nr_anims, dtype=int)
            animations = [self.animation_names[i] for i in idxes]
        else:
            animations = self.animation_names

        all_data = []
        for char in self.character_names:
            char_data = []
            for anim in animations:
                mot_dir = os.path.join(self.data_root, char, anim, 'motions')
                item = os.path.join(mot_dir, '1.npy')
                data, _ = self.preprocessing(item, aug=False)
                char_data.append(data)
            all_data.append(torch.stack(char_data, dim=0))

        all_data = torch.stack(all_data, dim=0)

        ret = (all_data, self.character_names, animations)
        return ret

    @staticmethod
    def augmentation(data, param=None):
        """
        :param data: numpy array of size (joints, 3, len_frames)
        :return:
        """
        if param is None:
            param = {}
            param['roll'] = np.random.uniform((-np.pi / 9, -np.pi / 9, -np.pi / 6), (np.pi / 9, np.pi / 9, np.pi / 6))
            param['ratio'] = np.random.uniform(0.8, 1.2)

        # rotate
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
        data = data * param['ratio']

        return data, param

    def preprocessing(self, item, aug, param=None):
        """
        :param data: numpy array of size (joints, 3, len_frames)
        :return:
        """
        motion3d = np.load(item)

        if aug:
            motion3d, param = self.augmentation(motion3d, param)

        motion_proj = trans_motion(motion3d)

        motion_proj = (motion_proj - self.mean_pose[:, :, np.newaxis]) / self.std_pose[:, :, np.newaxis]

        motion_proj = motion_proj.reshape((-1, motion_proj.shape[-1]))   # reshape to (joints*2, len_frames)

        motion_proj = torch.Tensor(motion_proj)

        return motion_proj, param

    def __getitem__(self, index):
        idx1, idx2 = np.random.choice(len(self.items), size=2)
        item1, item2 = self.items[idx1], self.items[idx2]
        item12, item21 = self.cross_2items(item1, item2)

        input1, param1 = self.preprocessing(item1, self.aug, None)
        input2, param2 = self.preprocessing(item2, self.aug, None)
        target1 = input1.detach().clone()
        target2 = input2.detach().clone()

        if self.aug:
            param12 = copy(param1)
            param21 = copy(param2)
            param12['ratio'] = param2['ratio']
            param21['ratio'] = param1['ratio']
        else:
            param12 = None
            param21 = None

        target12, _ = self.preprocessing(item12, self.aug, param12)
        target21, _ = self.preprocessing(item21, self.aug, param21)

        return {"input1": input1, "target1": target1,
                "input2": input2, "target2": target2,
                "target12": target12,
                "target21": target21,
                "name1": item1, "name2": item2,
                "name12": item12, "name21": item21}

    def __len__(self):
        return len(self.items)


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
    for char in character_names:
        char_dir = os.path.join(data_root, char)
        for anim in animation_names:
            mot_dir = os.path.join(char_dir, anim, 'motions')
            nr_motions = len(os.listdir(mot_dir))
            for i in range(nr_motions):
                motion3d = np.load(os.path.join(mot_dir, '{}.npy'.format(i + 1)))
                motion_proj = trans_motion(motion3d)

                all_joints.append(motion_proj)

    all_joints = np.concatenate(all_joints, axis=2)
    meanpose = np.mean(all_joints, axis=2)
    stdpose = np.std(all_joints, axis=2)
    stdpose[np.where(stdpose == 0)] = 1e-9
    np.save(MEAN_POSE_PATH, meanpose)
    np.save(STD_POSE_PATH, stdpose)


def get_meanpose():
    if not os.path.exists(MEAN_POSE_PATH):
        gen_meanpose(DATA_ROOT, TRAIN_CHARACTERS)

    mean_pose = np.load(MEAN_POSE_PATH)
    std_pose = np.load(STD_POSE_PATH)
    return mean_pose, std_pose


MEAN_POSE, STD_POSE = get_meanpose()


def test():
    mean_pose = get_meanpose() + np.array([256, 256]).reshape(1, 2)
    print(mean_pose.shape)
    train_ds = get_dataloaders('validation', batch_size=1)

    cluster_data = train_ds.dataset.get_cluster_data()
    print(cluster_data.shape)
    print(len(train_ds))
    for i, data in enumerate(train_ds):
        print(data['name1'][0])
        print(data['name2'][0])
        print(data['name12'][0])
        print(data['name21'][0])
        print(data['input1'][0].shape)
        break


if __name__=='__main__':
    gen_meanpose(DATA_ROOT, TRAIN_CHARACTERS)
    #test()
    meanpose, std_pose = get_meanpose()
    print(meanpose.shape, std_pose.shape)
    print(meanpose)
    print(std_pose)
    #valid_anim = get_valid_animations('/data1/wurundi/mixamo/validation', ['front'])
    #print(valid_anim)
    #print(len(valid_anim))