import os
import warnings
from common import config
import numpy as np
import torch
import argparse
from utils import Table
from tqdm import tqdm
from collections import OrderedDict
warnings.filterwarnings("ignore")
from dataset import DATA_ROOT, MEAN_POSE, STD_POSE, VALIDATION_CHARACTERS
from motion import *
import json


class EvaluateDataset(object):
    def __init__(self):
        self.data_root = DATA_ROOT

        self.character_names = VALIDATION_CHARACTERS

        self.animation_names = sorted(os.listdir(os.path.join(self.data_root, self.character_names[0])))

        self.items = []
        for char in self.character_names:
            for anim in self.animation_names:
                mot_dir = os.path.join(self.data_root, char, anim, 'motions')
                nr_motions = len(os.listdir(mot_dir))
                self.items.extend([os.path.join(mot_dir, '{}.npy'.format(i + 1))
                                   for i in range(nr_motions)])

        self.mean_pose, self.std_pose = MEAN_POSE, STD_POSE
        self.rng = np.random.RandomState(seed = 1024)

    def preprocessing(self, motion3d, is_target=True):

        motion_proj = trans_motion(motion3d)

        motion_proj = (motion_proj - self.mean_pose[:, :, np.newaxis]) / self.std_pose[:, :, np.newaxis]

        motion_proj = motion_proj.reshape((-1, motion_proj.shape[-1]))   # reshape to (joints*2, len_frames)

        if is_target:
            return motion_proj

        motion_proj = torch.Tensor(motion_proj).unsqueeze(0)

        return motion_proj

    def gen_motion(self, anim, char):
        json_dir = os.path.join(self.data_root, char, anim, 'jointsDict')
        json_list = sorted(os.listdir(json_dir))
        json_list = [os.path.join(json_dir, x) for x in json_list]
        mot = []
        for filename in json_list:
            with open(filename) as f:
                jointDict = json.load(f)
                pose_joints = np.array(jointDict['pose_keypoints_3d']).reshape((-1, 3))[:15]
                face_joints = np.array(jointDict['face_keypoints_3d']).reshape((-1, 3))
                joints = np.r_[pose_joints, face_joints]
                mot.append(joints)

        mot = np.stack(mot, axis=2)
        return mot

    def iter(self):
        for anim1 in self.animation_names:
            for char1 in self.character_names:
                available_anims = self.animation_names.copy()
                available_anims.remove(anim1)
                available_chars = self.character_names.copy()
                available_chars.remove(char1)

                anim2 = available_anims[self.rng.randint(0, len(available_anims))]
                char2 = available_chars[self.rng.randint(0, len(available_chars))]

                motion1 = self.gen_motion(anim1, char1)
                motion2 = self.gen_motion(anim2, char2)
                motion12 = self.gen_motion(anim1, char2)
                motion21 = self.gen_motion(anim2, char1)

                input1 = self.preprocessing(motion1, is_target=False)
                input2 = self.preprocessing(motion2, is_target=False)
                target1 = self.preprocessing(motion1)
                target2 = self.preprocessing(motion2)
                target12 = self.preprocessing(motion12)
                target21 = self.preprocessing(motion21)

                yield {"input1": input1, "target1": target1,
                        "input2": input2, "target2": target2,
                        "target12": target12,
                        "target21": target21,
                       'anim1': anim1, 'anim2': anim2,
                       'char1': char1, 'char2': char2}


def joint_L2_dist(motion1, motion2, each_joint=False):
    ret = np.mean(np.sqrt(np.sum((motion1 - motion2[:, :, :motion1.shape[-1]]) ** 2, axis=1)), axis=1)
    if each_joint:
        return ret
    return np.mean(ret)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str)
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_name = args.network.split('/')[-1].split('.')[0]

    net = torch.load(args.network)['net'].to(device)
    net.eval()

    val_dataset = EvaluateDataset()

    reconstruct_error = 0
    cross_error = 0
    all_error = 0
    joints_error = 0
    nr_samples = 0

    pbar = tqdm(val_dataset.iter())
    for data in pbar:
        input1 = data['input1'].to(device)
        input2 = data['input2'].to(device)
        targets = [data['target1'], data['target2'], data['target12'], data['target21']]

        outputs = net.cross(input1, input2)

        trans_outs = []
        for out in outputs:
            trans_outs.append(trans_motion_inv(normalize_motion_inv(out.detach().cpu().numpy()[0], MEAN_POSE, STD_POSE)))

        trans_targets = []
        for tgt in targets:
            trans_targets.append(trans_motion_inv(normalize_motion_inv(tgt, MEAN_POSE, STD_POSE)))

        assert len(trans_outs) == len(trans_targets)
        errors = [joint_L2_dist(trans_outs[i], trans_targets[i]) for i in range(len(trans_outs))]
        each_errors = [joint_L2_dist(trans_outs[i], trans_targets[i], each_joint=True) for i in range(len(trans_outs))]

        reconstruct_error += (errors[0] + errors[1]) / 2.0
        cross_error += (errors[2] + errors[3]) / 2.0
        all_error += np.mean(errors)
        joints_error += np.mean(np.stack(each_errors, axis=1), axis=1)
        nr_samples += 1

    reconstruct_error /= nr_samples
    cross_error /= nr_samples
    all_error /= nr_samples
    joints_error /= nr_samples

    stat_table = Table(config.stat_path)
    stat_info = OrderedDict({
        'name': config.exp_name,
        'model': model_name,
        'mot_layers': config.mot_en_channels,
        'body_layers': config.body_en_channels,
        'dec_layers': config.de_channels,
        'reconstruct_error': round(reconstruct_error, 6),
        'cross_error': round(cross_error, 6),
        'all_error': round(all_error, 6),
        'joint_error': [round(x, 6) for x in joints_error.tolist()]
    })
    stat_table.write(stat_info)

    selfout_path = os.path.join(config.exp_dir, 'eval-result-{}.txt'.format(stat_info['model']))

    for k, v in stat_info.items():
        print(k, ":", v)
        with open(selfout_path, 'w') as f:
            print(k, ":", v, file=f)
