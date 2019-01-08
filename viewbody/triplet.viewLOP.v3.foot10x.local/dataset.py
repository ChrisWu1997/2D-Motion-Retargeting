import os
from torch.utils.data import Dataset, DataLoader
# Ignore warnings
import warnings
from common import config
import numpy as np
import torch
from motion import trans_motion, trans_motion2d, rotate_coordinates
warnings.filterwarnings("ignore")

DATA_ROOT = '/data1/wurundi/mixamo/mixamo-3d-data'
TRAIN_CHARACTER = 'Jasper-full'
VALIDATION_CHARACTER = 'Jasper-full'
VIEW_RANGES = [(0, 0, -np.pi / 2),
               (0, 0, -np.pi / 3),
               (0, 0, -np.pi / 6),
               (0, 0, 0),
               (0, 0, np.pi / 6),
               (0, 0, np.pi / 3),
               (0, 0, np.pi / 2)]
MEAN_POSE_PATH = os.path.join('./model/meanpose_Jasper-full_7view-LOPv3-noroot.npy')
STD_POSE_PATH = os.path.join('./model/stdpose_Jasper-full_7view-LOPv3-noroot.npy')
np.random.seed(1234)
ANIMATIONS = np.random.permutation(sorted(os.listdir(os.path.join(DATA_ROOT, TRAIN_CHARACTER))))
NR_VAL = 100

class MixamoDataset(Dataset):
    def __init__(self, name):
        super(MixamoDataset, self).__init__()
        self.data_root = DATA_ROOT

        self.character_name = TRAIN_CHARACTER

        self.animation_names = ANIMATIONS

        if name == 'train':
            self.animation_names = self.animation_names[:-NR_VAL]
            self.aug = True
        elif name == 'validation':
            self.animation_names = self.animation_names[-NR_VAL:]
            self.aug = False
        else:
            raise NameError

        self.view_angles = VIEW_RANGES
        self.items = []
        for anim in self.animation_names:
            mot_dir = os.path.join(self.data_root, self.character_name, anim, 'motions')
            local_dir = os.path.join(self.data_root, self.character_name, anim, 'locals')
            nr_motions = len(os.listdir(mot_dir))
            self.items.extend([(os.path.join(mot_dir, '{}.npy'.format(i + 1)),
                                os.path.join(local_dir, '{}.npy'.format(i + 1)))
                                for i in range(nr_motions)])

        self.mean_pose, self.std_pose = get_meanpose()

    def get_cluster_data(self, nr_anims=100):
        """
        get a certain subset data for clustering visualization and scoring
        :param nr_anims:
        :return: pre-processed data of shape (nr_views, nr_anims, 30, 64)
        """
        animations = sorted(self.animation_names)
        if nr_anims < len(animations):
            idxes = np.linspace(0, len(animations) - 1, nr_anims, dtype=int)
            animations = [animations[i] for i in idxes]

        all_data = []
        for view in self.view_angles:
            view_data = []
            for anim in animations:
                mot_dir = os.path.join(self.data_root, self.character_name, anim, 'motions')
                local_dir = os.path.join(self.data_root, self.character_name, anim, 'locals')
                item = (os.path.join(mot_dir, '1.npy'), os.path.join(local_dir, '1.npy'))
                data, _ = self.preprocessing(item, view, aug=False)
                view_data.append(data)
            all_data.append(torch.stack(view_data, dim=0))

        all_data = torch.stack(all_data, dim=0)

        ret = (all_data, np.rad2deg(self.view_angles), animations)
        return ret

    @staticmethod
    def augmentation(data, param=None):
        """
        :param data: numpy array of size (joints, 3, len_frames)
        :return:
        """
        '''
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
        data = mat33_x @ mat33_y @ data

        # scale
        data = data * param['ratio']

        return data, param
        '''
        pass

    def preprocessing(self, item, view_angle, aug, param=None):
        """
        :param data: numpy array of size (joints, 3, len_frames)
        :return:
        """
        motion3d = np.load(item[0])
        local3d = np.load(item[1])

        motion3d = motion3d[:15, :, :]

        if aug:
            motion3d = motion3d * param['scale']

        # convert 3d to 2d
        local3d = rotate_coordinates(local3d, view_angle)

        motion_proj, centers = trans_motion(motion3d, local3d)

        motion_proj = (motion_proj - self.mean_pose[:, :, np.newaxis]) / self.std_pose[:, :, np.newaxis]
        motion_proj = motion_proj.reshape((-1, motion_proj.shape[-1]))   # reshape to (joints*2, len_frames)

        motion_proj = torch.Tensor(motion_proj)
        centers = torch.Tensor(centers)

        return motion_proj, centers

    def __getitem__(self, index):
        idx1, idx2 = np.random.choice(len(self.items), size=2, replace=False)
        item1, item2 = self.items[idx1], self.items[idx2]
        idx1, idx2 = np.random.choice(len(self.view_angles), size=2, replace=False)
        view1, view2 = self.view_angles[idx1], self.view_angles[idx2]

        param = None
        if self.aug:
            param = {}
            param['scale'] = np.random.uniform(0.8, 1.2)
        input1, center1 = self.preprocessing(item1, view1, self.aug, param)
        input2, center2 = self.preprocessing(item2, view2, self.aug, param)
        target1 = input1.detach().clone()
        target2 = input2.detach().clone()

        input12, center12 = self.preprocessing(item1, view2, self.aug, param)
        input21, center21 = self.preprocessing(item2, view1, self.aug, param)

        input12.requires_grad = False
        input21.requires_grad = False
        target12 = input12.detach().clone()
        target21 = input21.detach().clone()

        return {"input1": input1, "target1": target1, "center1": center1,
                "input2": input2, "target2": target2, "center2": center2,
                "input12": input12, "target12": target12, "center12": center12,
                "input21": input21, "target21": target21, "center21": center21,
                "name1": item1, "name2": item2,
                "view1": view1, "view2": view2}

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


def gen_meanpose(data_root, character_name):
    animation_names = os.listdir(os.path.join(data_root, character_name))

    all_joints = []
    char_dir = os.path.join(data_root, character_name)
    for anim in animation_names:
        mot_dir = os.path.join(char_dir, anim, 'motions')
        local_dir = os.path.join(char_dir, anim, 'locals')
        nr_motions = len(os.listdir(mot_dir))
        for i in range(nr_motions):
            motion3d = np.load(os.path.join(mot_dir, '{}.npy'.format(i + 1)))
            local3d = np.load(os.path.join(local_dir, '{}.npy'.format(i + 1)))

            for view in VIEW_RANGES:
                local3d = rotate_coordinates(local3d, view)
                motion_proj, centers = trans_motion(motion3d, local3d)
                all_joints.append(motion_proj)

    all_joints = np.concatenate(all_joints, axis=2)
    meanpose = np.mean(all_joints, axis=2)
    stdpose = np.std(all_joints, axis=2)
    stdpose[np.where(stdpose == 0)] = 1e-9
    np.save(MEAN_POSE_PATH, meanpose)
    np.save(STD_POSE_PATH, stdpose)


def get_meanpose():
    if not os.path.exists(MEAN_POSE_PATH):
        gen_meanpose(DATA_ROOT, TRAIN_CHARACTER)

    mean_pose = np.load(MEAN_POSE_PATH)
    std_pose = np.load(STD_POSE_PATH)
    return mean_pose, std_pose


MEAN_POSE, STD_POSE = get_meanpose()


def test():
    mean_pose = get_meanpose() + np.array([256, 256]).reshape(1, 2)
    print(mean_pose.shape)
    train_ds = get_dataloaders('train', batch_size=1)

    cluster_data = train_ds.dataset.get_cluster_data()
    print(cluster_data[0].shape)
    print(len(train_ds))

    val_ds = get_dataloaders('validation', batch_size=1)
    print(len(val_ds))
    for i, data in enumerate(train_ds):
        print(data['name1'][0])
        print(data['name2'][0])
        print(data['view1'][0])
        print(data['view2'][0])
        print(data['input1'][0].shape)
        break


if __name__=='__main__':
    meanpose, std_pose = get_meanpose()
    print(meanpose.shape, std_pose.shape)
    print(meanpose)
    print(std_pose)