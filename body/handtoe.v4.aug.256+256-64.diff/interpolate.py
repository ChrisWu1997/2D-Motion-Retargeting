import os
from common import config
import numpy as np
import torch
import argparse
import json
import torch.nn.functional as F
from tqdm import tqdm
from visulization import pose2im_all, joints2image
import cv2
import imageio
from dataset import MEAN_POSE, STD_POSE
from motion import normalize_motion, normalize_motion_inv, trans_motion_inv, trans_motion2d
from scipy.ndimage import gaussian_filter1d
import time
from utils import ensure_dir


def get_data(json_dir, is_mixamo, scale=1.0):
    paths = sorted(os.listdir(json_dir))
    paths = [os.path.join(json_dir, x) for x in paths]

    def json2motion(json_list):
        mot = []
        for filename in json_list:
            with open(filename) as f:
                jointDict = json.load(f)
                if is_mixamo:
                    pose_joints = np.array(jointDict['pose_keypoints_3d']).reshape((-1, 3))[:15]
                    face_joints = np.array(jointDict['face_keypoints_3d']).reshape((-1, 3))
                    joint = np.r_[pose_joints, face_joints]
                else:
                    joint = np.array(jointDict['people'][0]['pose_keypoints_2d']).reshape((-1, 3))[:17, :2]
                if len(mot) > 0:
                    joint[np.where(joint == 0)] = mot[-1][np.where(joint == 0)]
                mot.append(joint)

        mot = np.stack(mot, axis=2)
        mot = gaussian_filter1d(mot, sigma=2, axis=-1)
        if is_mixamo:
            mot = mot[:, [0, 2], :]  # (15, 2, 64)
            mot[:, 1, :] = - mot[:, 1, :]

            mot = mot * config.unit

            mot[1, :, :] = (mot[2, :, :] + mot[5, :, :]) / 2
            mot[8, :, :] = (mot[9, :, :] + mot[12, :, :]) / 2

            centers = mot[8, :, :]
            mot = mot - centers
            centers_reset = centers - centers[:, 0].reshape(2, 1) \
                            + np.array([[config.img_size[0] // 2], [config.img_size[1] // 2]])
            mot = mot + centers_reset
        return mot

    def preprocessing(motion):
        motion_proj = trans_motion2d(motion)

        motion_proj = normalize_motion(motion_proj, MEAN_POSE, STD_POSE)
        motion_proj = motion_proj.reshape((-1, motion_proj.shape[-1]))  # reshape to (joints*2, len_frames)

        motion_proj = torch.Tensor(motion_proj).unsqueeze(0)

        return motion_proj

    motion = json2motion(paths)
    motion = motion * scale
    motion_input = preprocessing(motion)

    data = {'input': motion_input, 'raw': motion}
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network', type=str)
    parser.add_argument('-v1', '--vid1_json_dir', type=str)
    parser.add_argument('-v2', '--vid2_json_dir', type=str)
    parser.add_argument('-h1', '--img1_height', type=int)
    parser.add_argument('-h2', '--img2_height', type=int)
    parser.add_argument('-w1', '--img1_width', type=int)
    parser.add_argument('-w2', '--img2_width', type=int)
    parser.add_argument("--mixamo1", help='if v1 belongs to mixamo', action="store_true")
    parser.add_argument("--mixamo2", help='if v2 belongs to mixamo', action="store_true")
    parser.add_argument('-o', '--out_path', default='./predict.mp4', type=str)
    parser.add_argument('--write_json', type=str)
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False)
    args = parser.parse_args()


    def pad_to_16x(x):
        if x % 16 > 0:
            return x - x % 16 + 16
        return x


    h1, w1 = args.img1_height, args.img1_width
    h2, w2 = args.img2_height, args.img2_width
    scale1 = config.img_size[0] / h1
    scale2 = config.img_size[0] / h2
    h1 = h2 = config.img_size[0]
    w1 = pad_to_16x(int(w1 * scale1))
    w2 = pad_to_16x(int(w2 * scale2))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = torch.load(args.network)['net'].to(device)
    net.eval()

    data1 = get_data(args.vid1_json_dir, args.mixamo1, scale1)
    data2 = get_data(args.vid2_json_dir, args.mixamo2, scale2)

    since = time.time()

    input1 = data1['input'].to(device)
    input2 = data2['input'].to(device)
    vlen = min(input1.shape[-1], input2.shape[-1]) // 16 * 16
    print(vlen)
    input1 = input1[:, :, :vlen]
    input2 = input2[:, :, :vlen]

    # interpolation
    m1 = net.mot_encoder(input1)
    m2 = net.mot_encoder(input2)
    b1 = net.body_encoder(input1[:, :-2, :])
    b2 = net.body_encoder(input2[:, :-2, :])

    alphas = torch.linspace(0, 1, vlen).to(device)
    b1_mix = torch.cat([(1 - alpha) * b1 + alpha * b2 for alpha in alphas], dim=0)
    b2_mix = torch.cat([alpha * b1 + (1 - alpha) * b2 for alpha in alphas], dim=0)
    m1 = m1.repeat(vlen, 1, 1)
    m2 = m2.repeat(vlen, 1, 1)
    out12 = net.decoder(torch.cat([m1, b1_mix.repeat(1, 1, m1.shape[-1])], dim=1))
    out21 = net.decoder(torch.cat([m2, b2_mix.repeat(1, 1, m2.shape[-1])], dim=1))

    idx = torch.arange(vlen).to(device)
    out12 = out12[idx, :, idx].transpose(1, 0)
    out21 = out21[idx, :, idx].transpose(1, 0)

    input1 = input1.detach().cpu().numpy()[0].reshape(-1, 2, input1.shape[-1])
    input2 = input2.detach().cpu().numpy()[0].reshape(-1, 2, input2.shape[-1])
    out12 = out12.detach().cpu().numpy().reshape(-1, 2, out12.shape[-1])
    out21 = out21.detach().cpu().numpy().reshape(-1, 2, out21.shape[-1])

    input1 = trans_motion_inv(normalize_motion_inv(input1, MEAN_POSE, STD_POSE), w1 // 2, h1 // 2)
    input2 = trans_motion_inv(normalize_motion_inv(input2, MEAN_POSE, STD_POSE), w2 // 2, h2 // 2)
    out12 = trans_motion_inv(normalize_motion_inv(out12, MEAN_POSE, STD_POSE), w2 // 2, h2 // 2)
    out21 = trans_motion_inv(normalize_motion_inv(out21, MEAN_POSE, STD_POSE), w1 // 2, h1 // 2)

    total_time = time.time() - since
    print("total time:", total_time)
    print("1/fps:", total_time / max(input1.shape[-1], input2.shape[-1]))


    def motion2json(motion, h, w, save_dir):
        nr_frames = motion.shape[-1]
        for i in range(nr_frames):
            path = os.path.join(save_dir, "%04d_keypoints.json" % i)
            out_dict = {"resolution": (h, w), 'pose_keypoints': motion[:, :, i].reshape(-1).tolist()}
            with open(path, 'w') as f:
                json.dump(out_dict, f)


    if args.write_json is not None:
        ensure_dir(args.write_json)
        out1_dir = os.path.join(args.write_json, 'point-input1')
        out2_dir = os.path.join(args.write_json, 'point-input2')
        out12_dir = os.path.join(args.write_json, 'point-motion1char2view2')
        out21_dir = os.path.join(args.write_json, 'point-motion2char1view1')
        ensure_dir(out1_dir)
        ensure_dir(out2_dir)
        ensure_dir(out12_dir)
        ensure_dir(out21_dir)

        motion2json(input1, h1, w1, out1_dir)
        motion2json(input2, h2, w2, out2_dir)
        motion2json(out12, h2, w2, out12_dir)
        motion2json(out21, h1, w1, out21_dir)

    out_path = os.path.join(args.write_json, 'compare.mp4') if args.write_json is not None else args.out_path
    videowriter = imageio.get_writer(out_path, fps=25)
    vlen = min(input1.shape[-1], input2.shape[-1])

    for i in tqdm(range(vlen)):
        img_1 = joints2image(input1[:, :, i], H=h1, W=w1) if i < input1.shape[-1] else np.zeros((h1, h1, 3),
                                                                                               dtype=np.float)
        img_2 = joints2image(input2[:, :, i], H=h2, W=w2) if i < input2.shape[-1] else np.zeros((h2, h2, 3),
                                                                                               dtype=np.float)
        img_out12 = joints2image(out12[:, :, i], H=h1, W=w2) if i < out12.shape[-1] else np.zeros((h1, h1, 3),
                                                                                                 dtype=np.float)
        img_out21 = joints2image(out21[:, :, i], H=h2, W=w1) if i < out21.shape[-1] else np.zeros((h2, h2, 3),
                                                                                                 dtype=np.float)

        cv2.putText(img_1, 'input1', (20, 20), 1, 2, (0, 0, 255), 1)
        cv2.putText(img_2, 'input2', (20, 20), 1, 2, (0, 0, 255), 1)
        cv2.putText(img_out12, 'motion1+char2+view2', (20, 20), 1, 2, (0, 0, 255), 1)
        cv2.putText(img_out21, 'motion2+char1+view1', (20, 20), 1, 2, (0, 0, 255), 1)

        up_img = np.concatenate([img_1, img_2], axis=1)
        down_img = np.concatenate([img_out12, img_out21], axis=1)
        whole_img = np.concatenate([up_img, down_img], axis=0)
        #whole_img = np.concatenate([img_1, img_2, img_out12, img_out21], axis=1)
        videowriter.append_data(whole_img)
    videowriter.close()
