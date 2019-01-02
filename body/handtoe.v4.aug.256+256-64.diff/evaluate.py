import os
from common import config
import numpy as np
import torch
import argparse
import json
import torch.nn.functional as F
from utils import AverageMeter, Table
from tqdm import tqdm
from collections import OrderedDict
from dataset import DATA_ROOT, VALIDATION_CHARACTERS, MEAN_POSE, STD_POSE
from motion import trans_motion, trans_motion_inv, normalize_motion, normalize_motion_inv


def data_iter():
    data_root = DATA_ROOT

    character_names = VALIDATION_CHARACTERS

    animation_names = os.listdir(os.path.join(data_root, character_names[0]))

    print('number of characters:', len(character_names))
    print('number of animations:', len(animation_names))

    def gen_motion(joint_dir):
        nr_files = len(os.listdir(joint_dir))
        mot = []
        for i in range(0, nr_files):
            with open(os.path.join(joint_dir, '%04d_keypoints.json' % i)) as f:
                joint = json.load(f)
                joint = np.array(joint['people'][0]['pose_keypoints_2d']).reshape((-1, 3))
                mot.append(joint)

        mot = np.stack(mot, axis=2)
        return mot

    def preprocessing(motion3d, is_target=False):
        motion3d = motion3d[:15, :, :]

        motion_proj = trans_motion(motion3d, is_target)
        if is_target:
            return motion_proj

        motion_proj = normalize_motion(motion_proj, MEAN_POSE, STD_POSE)
        motion_proj = motion_proj.reshape((-1, motion_proj.shape[-1]))

        motion_proj = torch.Tensor(motion_proj).unsqueeze(0)

        return motion_proj

    np.random.seed(1024)
    for char1 in character_names:
        for char2 in character_names:
            if char2 == char1:
                continue
            for anim in animation_names:
                joint_dir1 = os.path.join(data_root, char1, anim, 'jointsDict')
                mot1_inp = gen_motion(joint_dir1)
                mot1_inp = preprocessing(mot1_inp)

                joint_dir2 = os.path.join(data_root, char2, anim, 'jointsDict')
                mot2_tgt = gen_motion(joint_dir2)
                mot2_tgt = preprocessing(mot2_tgt, is_target=True)

                ref = np.random.choice(animation_names)
                joint_ref_dir = os.path.join(data_root, char2, ref, 'jointsDict')
                mot2_inp = gen_motion(joint_ref_dir)
                mot2_inp = preprocessing(mot2_inp)

                data = {'input1': mot1_inp, 'input2': mot2_inp, 'target': mot2_tgt,
                        'item1': '{}.{}'.format(char1, anim),
                        'item2': '{}.{}'.format(char2, ref)}
                yield data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str)
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_name = args.network.split('/')[-1].split('.')[0]
    eval_log_path = os.path.join(config.exp_dir, 'eval-log-{}.txt'.format(model_name))

    net = torch.load(args.network)['net'].to(device)
    net.eval()

    metrics = AverageMeter('losses')
    pbar = tqdm(data_iter())
    for data in pbar:
        input1 = data['input1'].to(device)
        input2 = data['input2'].to(device)
        target = data['target']
        item1 = data['item1']
        item2 = data['item2']

        output = net.transfer(input1, input2)
        output = output.detach().cpu().numpy()[0].reshape(-1, 2, output.shape[-1])
        output = trans_motion_inv(normalize_motion_inv(output, MEAN_POSE, STD_POSE))

        l2_joint_dist = np.mean(np.mean(np.sqrt(np.sum((output - target[:, :, :output.shape[-1]]) ** 2, axis=1)), axis=1))

        metrics.update(l2_joint_dist)

        with open(eval_log_path, 'a') as f:
            print('{}---{}: {:4f}'.format(item1, item2, l2_joint_dist), file=f)

    model_name = args.network.split('/')[-1].split('.')[0]
    stat_table = Table(config.stat_path)
    stat_info = OrderedDict({
        'name': config.exp_name,
        'model': model_name,
        'mot_layers': config.mot_en_channels,
        'body_layers': config.body_en_channels,
        'dec_layers': config.de_channels,
        'augmentation': '3d rotaion+scale',
        'global motion': 'velocity, remove root row',
        'structure': 'v2, ks=8, layer128, mean & std',
        'train data': '16709 motions 11 chars fbxjoints',
        'val data': 'maximo combat 5 chars 920 transfer',
        'avg l2 joint dist': round(metrics.avg, 6),
    })
    stat_table.write(stat_info)

    selfout_path = os.path.join(config.exp_dir, 'eval-result-{}.txt'.format(stat_info['model']))

    for k, v in stat_info.items():
        print(k, ":", v)
        with open(selfout_path, 'w') as f:
            print(k, ":", v, file=f)
