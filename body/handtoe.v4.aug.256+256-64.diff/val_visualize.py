import os
from common import config
import numpy as np
import torch
import argparse
import json
from tqdm import tqdm
from visulization import joints2image
import cv2
import imageio
from dataset import MEAN_POSE, STD_POSE
from utils import ensure_dir
from motion import *
from scipy.ndimage import gaussian_filter1d


def get_data(vid1_json_dir, vid2_json_dir):

    def get_json_list(json_dir):
        json_list = sorted(os.listdir(json_dir))
        json_list = [os.path.join(json_dir, x) for x in json_list]
        return json_list

    def cross_two(path1, path2):
        l1 = path1.split('/')
        l2 = path2.split('/')
        mot1, char1 = l1[-2], l1[-3]
        mot2, char2 = l2[-2], l2[-3]
        l1[-3] = char2
        l2[-3] = char1
        path12 = os.path.join('/', *l1)
        path21 = os.path.join('/', *l2)
        return path12, path21

    vid12_json_dir, vid21_json_dir = cross_two(vid1_json_dir, vid2_json_dir)

    vid1_json_list = get_json_list(vid1_json_dir)
    vid2_json_list = get_json_list(vid2_json_dir)
    vid12_json_list = get_json_list(vid12_json_dir)
    vid21_json_list = get_json_list(vid21_json_dir)

    def gen_motion(json_list):
        mot = []
        for filename in json_list:
            with open(filename) as f:
                jointDict = json.load(f)
                pose_joints = np.array(jointDict['pose_keypoints_3d']).reshape((-1, 3))[:17, :]
                face_joints = np.array(jointDict['face_keypoints_3d']).reshape((-1, 3))
                hand_joints = np.array(jointDict['hand_keypoints_3d']).reshape((-1, 3))
                joints = np.r_[pose_joints, face_joints, hand_joints]
                mot.append(joints)

        mot = np.stack(mot, axis=2)
        return mot

    def preprocessing(motion3d, is_target=False):

        motion_proj = trans_motion(motion3d, is_target)

        motion_proj = normalize_motion(motion_proj, MEAN_POSE, STD_POSE)
        motion_proj = motion_proj.reshape((-1, motion_proj.shape[-1]))

        if is_target:
            return motion_proj
        motion_proj = torch.Tensor(motion_proj).unsqueeze(0)

        return motion_proj

    inp1 = gen_motion(vid1_json_list)
    inp1 = preprocessing(inp1)
    inp2 = gen_motion(vid2_json_list)
    inp2 = preprocessing(inp2)

    tgt12 = gen_motion(vid12_json_list)
    tgt21 = gen_motion(vid21_json_list)
    tgt12 = preprocessing(tgt12, is_target=True)
    tgt21 = preprocessing(tgt21, is_target=True)

    data = {'input1': inp1, 'input2': inp2,
            'target12': tgt12, 'target21': tgt21,
            }

    return data


def motion2video(motion, h, w, save_path, motion_tgt=None):
    videowriter = imageio.get_writer(save_path, fps=25)
    vlen = motion.shape[-1]
    for i in tqdm(range(vlen)):
        img = joints2image(motion[:, :, i], H=h, W=w)
        if motion_tgt is not None:
            img_tgt = joints2image(motion_tgt[:, :, i], H=h, W=w)
            img = cv2.addWeighted(img_tgt, 0.3, img, 0.7, 0)
        videowriter.append_data(img)
    videowriter.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network', type=str)
    parser.add_argument('-v1', '--vid1_json_dir', type=str)
    parser.add_argument('-v2', '--vid2_json_dir', type=str)
    parser.add_argument('-h1', '--img1_height', default=512, type=int)
    parser.add_argument('-h2', '--img2_height', default=512, type=int)
    parser.add_argument('-o', '--save_dir', type=str)
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False)
    parser.add_argument('-u', '--unit', type=int, default=128)
    args = parser.parse_args()

    h = w = config.img_size[0]
    unit = args.unit

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = torch.load(args.network)['net'].to(device)
    net.eval()

    data = get_data(args.vid1_json_dir, args.vid2_json_dir)

    input1 = data['input1'].to(device)
    input2 = data['input2'].to(device)
    target12 = data['target12']
    target21 = data['target21']

    out12 = net.transfer(input1, input2)
    out21 = net.transfer(input2, input1)

    input1 = input1.detach().cpu().numpy()[0].reshape(-1, 2, input1.shape[-1])
    input2 = input2.detach().cpu().numpy()[0].reshape(-1, 2, input2.shape[-1])
    out12 = out12.detach().cpu().numpy()[0].reshape(-1, 2, out12.shape[-1])
    out21 = out21.detach().cpu().numpy()[0].reshape(-1, 2, out21.shape[-1])

    input1 = trans_motion_inv(normalize_motion_inv(input1, MEAN_POSE, STD_POSE))
    input2 = trans_motion_inv(normalize_motion_inv(input2, MEAN_POSE, STD_POSE))
    out12 = trans_motion_inv(normalize_motion_inv(out12, MEAN_POSE, STD_POSE))
    out21 = trans_motion_inv(normalize_motion_inv(out21, MEAN_POSE, STD_POSE))
    target12 = trans_motion_inv(normalize_motion_inv(target12, MEAN_POSE, STD_POSE))
    target21 = trans_motion_inv(normalize_motion_inv(target21, MEAN_POSE, STD_POSE))

    # post-processing
    out12 = gaussian_filter1d(out12, sigma=4)
    out21 = gaussian_filter1d(out21, sigma=4)

    ensure_dir(args.save_dir)
    motion2video(input1, h, w, os.path.join(args.save_dir, 'input1.mp4'))
    motion2video(input2, h, w, os.path.join(args.save_dir, 'input2.mp4'))
    motion2video(out12, h, w, os.path.join(args.save_dir, 'output12.mp4'), target12)
    motion2video(out21, h, w, os.path.join(args.save_dir, 'target21.mp4'), target21)

