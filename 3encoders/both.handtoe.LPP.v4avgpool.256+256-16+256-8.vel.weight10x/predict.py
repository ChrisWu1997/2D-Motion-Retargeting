import os
import warnings
from common import config
import numpy as np
import torch
import argparse
import json
from tqdm import tqdm
warnings.filterwarnings("ignore")
from visulization import pose2im_all, joints2image, hex2rgb
import cv2
import imageio
from dataset import MEAN_POSE, STD_POSE
from motion import *
from scipy.ndimage import gaussian_filter1d
from utils import ensure_dir


# used for openpose format
def get_data(json_dir, is_mixamo, scale=1.0):
    def preprocessing(motion):
        if is_mixamo:
            motion_proj = trans_motion(motion)
        else:
            motion_proj = trans_motion2d(motion)

        motion_proj = normalize_motion(motion_proj, MEAN_POSE, STD_POSE)
        motion_proj = motion_proj.reshape((-1, motion_proj.shape[-1]))  # reshape to (joints*2, len_frames)

        motion_proj = torch.Tensor(motion_proj).unsqueeze(0)

        return motion_proj

    motion = json2motion(json_dir, scale, is_mixamo=is_mixamo)
    #motion = motion * scale
    motion_input = preprocessing(motion)

    data = {'input': motion_input, 'raw': motion}
    return data


def motion2video(motion, h, w, save_path, colors, motion_tgt=None):
    videowriter = imageio.get_writer(save_path, fps=25)
    vlen = motion.shape[-1]
    for i in tqdm(range(vlen)):
        img = joints2image(motion[:, :, i], colors, H=h, W=w)
        if motion_tgt is not None:
            img_tgt = joints2image(motion_tgt[:, :, i], colors, H=h, W=w)
            img = cv2.addWeighted(img_tgt, 0.3, img, 0.7, 0)
        videowriter.append_data(img)
    videowriter.close()


def motion2json(motion, h, w, save_dir):
    nr_frames = motion.shape[-1]
    for i in range(nr_frames):
        path = os.path.join(save_dir, "%04d_keypoints.json" % i)
        out_dict = {"resolution": (h, w), 'pose_keypoints': motion[:, :, i].reshape(-1).tolist()}
        with open(path, 'w') as f:
            json.dump(out_dict, f)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network', type=str)
    parser.add_argument('-v1', '--vid1_json_dir', type=str)
    parser.add_argument('-v2', '--vid2_json_dir', type=str)
    parser.add_argument('-h1', '--img1_height', type=int)
    parser.add_argument('-h2', '--img2_height', type=int)
    parser.add_argument('-w1', '--img1_width', type=int)
    parser.add_argument('-w2', '--img2_width', type=int)
    parser.add_argument('-v3', '--vid3_json_dir', type=str)
    parser.add_argument('-h3', '--img3_height', type=int)
    parser.add_argument('-w3', '--img3_width', type=int)
    parser.add_argument("--mixamo1", help='if v1 belongs to mixamo', action="store_true")
    parser.add_argument("--mixamo2", help='if v2 belongs to mixamo', action="store_true")
    parser.add_argument("--mixamo3", help='if v3 belongs to mixamo', action="store_true")
    parser.add_argument('--color1', type=str, default='#ff0000##aa0000#550000', help='color1')
    parser.add_argument('--color2', type=str, default='#0000ff#0000aa#000055', help='color2')
    parser.add_argument('--color3', type=str, default='#00ff00#00aa00#005500', help='color2')
    parser.add_argument('--mode', type=str, choices=['body', 'view', 'both'], default='both')
    parser.add_argument('--smooth', action='store_true',
                        help="to smooth the output using gaussian kernel")
    parser.add_argument('--max_length', type=int, default=240,
                        help='maximum input video length')
    parser.add_argument('--use_tgt_vel', action='store_true',
                        help="to use the target input's velocity(difference) for output results if set")
    parser.add_argument('-o', '--save_dir', type=str,
                        help='the folder folder to write the output videos')
    parser.add_argument('--write_json', type=str,
                        help='the folder to write output json')
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

    input1 = data1['input'].to(device)
    input2 = data2['input'].to(device)
    vlen = min(input1.shape[-1], input2.shape[-1], args.max_length) // 16 * 16
    input1 = input1[:, :, :vlen]
    input2 = input2[:, :, :vlen]

    if args.vid3_json_dir is not None:
        h3, w3 = args.img3_height, args.img3_width
        scale3 = config.img_size[0] / h3
        w3 = pad_to_16x(int(w3 * scale3))
        data3 = get_data(args.vid3_json_dir, args.mixamo3, scale3)
        input3 = data3['input'].to(device)
        input3 = input3[:, :, :vlen]
        out12 = net.transfer_three(input1, input2, input3)
        out21 = net.transfer_three(input3, input2, input1)
    elif args.mode == 'body':
        out12 = net.transfer_body(input1, input2)
        out21 = net.transfer_body(input2, input1)
    elif args.mode == 'view':
        out12 = net.transfer_view(input1, input2)
        out21 = net.transfer_view(input2, input1)
    else:
        out12 = net.transfer_both(input1, input2)
        out21 = net.transfer_both(input2, input1)

    input1 = input1.detach().cpu().numpy()[0]
    input2 = input2.detach().cpu().numpy()[0]
    out12 = out12.detach().cpu().numpy()[0]
    out21 = out21.detach().cpu().numpy()[0]

    input1 = normalize_motion_inv(input1, MEAN_POSE, STD_POSE)
    input2 = normalize_motion_inv(input2, MEAN_POSE, STD_POSE)
    out12 = normalize_motion_inv(out12, MEAN_POSE, STD_POSE)
    out21 = normalize_motion_inv(out21, MEAN_POSE, STD_POSE)
    if args.use_tgt_vel:
        out12 = trans_motion_inv(out12, w2 // 2, h2 // 2, input1[-1].copy())
        out21 = trans_motion_inv(out21, w1 // 2, h1 // 2, input2[-1].copy())
    else:
        out12 = trans_motion_inv(out12, w2 // 2, h2 // 2)
        out21 = trans_motion_inv(out21, w1 // 2, h1 // 2)

    input1 = trans_motion_inv(input1, w1 // 2, h1 // 2)
    input2 = trans_motion_inv(input2, w2 // 2, h2 // 2)
    if args.vid3_json_dir is not None:
        input3 = input3.detach().cpu().numpy()[0]
        input3 = normalize_motion_inv(input3, MEAN_POSE, STD_POSE)
        input3 = trans_motion_inv(input3, w3 // 2, h3 // 2)

    if args.smooth:
        out12 = gaussian_filter1d(out12, sigma=2, axis=-1)
        out21 = gaussian_filter1d(out21, sigma=2, axis=-1)

    if args.write_json is not None:
        ensure_dir(args.write_json)
        out1_dir = os.path.join(args.write_json, 'point-input1')
        out2_dir = os.path.join(args.write_json, 'point-input2')
        out12_dir = os.path.join(args.write_json, 'point-output12')
        out21_dir = os.path.join(args.write_json, 'point-output21')
        ensure_dir(out1_dir)
        ensure_dir(out2_dir)
        ensure_dir(out12_dir)
        ensure_dir(out21_dir)
        motion2json(input1, h1, w1, out1_dir)
        motion2json(input2, h2, w2, out2_dir)
        motion2json(out12, h2, w2, out12_dir)
        motion2json(out21, h1, w1, out21_dir)

        print('Json files are writen.')

    if args.save_dir is not None or args.write_json is not None:
        save_dir = args.write_json if args.write_json is not None else args.save_dir
        ensure_dir(save_dir)
        color1 = hex2rgb(args.color1)
        color2 = hex2rgb(args.color2)
        motion2video(input1, h1, w1, os.path.join(save_dir, 'input1.mp4'), color1)
        motion2video(input2, h2, w2, os.path.join(save_dir, 'input2.mp4'), color2)
        if args.vid3_json_dir is not None:
            color3 = hex2rgb(args.color3)
            motion2video(input3, h3, w3, os.path.join(save_dir, 'input3.mp4'), color3)
        motion2video(out12, h2, w2, os.path.join(save_dir, 'output12.mp4'), color2)
        motion2video(out21, h1, w1, os.path.join(save_dir, 'output21.mp4'), color1)
