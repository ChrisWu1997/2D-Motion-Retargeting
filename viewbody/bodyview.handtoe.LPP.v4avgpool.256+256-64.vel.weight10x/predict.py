import os
from common import config
import numpy as np
import torch
import argparse
import json
import torch.nn.functional as F
from tqdm import tqdm
from visualization import pose2im_all, joints2image
import cv2
import imageio
from dataset import MEAN_POSE, STD_POSE
from motion import normalize_motion, normalize_motion_inv, trans_motion_inv, trans_motion2d, json2motion, trans_motion
from scipy.ndimage import gaussian_filter1d
import time
from utils import ensure_dir
from utils import save_image


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


def motion2video(motion, h, w, save_path, name, colors, transparency=False, motion_tgt=None, fps=25):
    videowriter = imageio.get_writer(os.path.join(save_path, name + '.mp4'), fps=fps)
    vlen = motion.shape[-1]
    frames_dir = os.path.join(save_path, 'frames-' + name)
    ensure_dir(frames_dir)
    for i in tqdm(range(vlen)):
        [img, img_cropped] = joints2image(motion[:, :, i], colors, transparency, H=h, W=w)
        if motion_tgt is not None:
            [img_tgt, img_tgt_cropped] = joints2image(motion_tgt[:, :, i], colors, H=h, W=w)
            img = cv2.addWeighted(img_tgt, 0.3, img, 0.7, 0)
        save_image(img_cropped, os.path.join(frames_dir, "%04d.png" % i))
        videowriter.append_data(img)
    videowriter.close()


def motion2json(motion, h, w, save_dir):
    nr_frames = motion.shape[-1]
    for i in range(nr_frames):
        path = os.path.join(save_dir, "%04d_keypoints.json" % i)
        out_dict = {"resolution": (h, w), 'pose_keypoints': motion[:, :, i].reshape(-1).tolist()}
        with open(path, 'w') as f:
            json.dump(out_dict, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network', type=str)
    parser.add_argument('-v1', '--vid1_json_dir', type=str)
    parser.add_argument('-v2', '--vid2_json_dir', type=str)
    parser.add_argument('-h1', '--img1_height', type=int)
    parser.add_argument('-h2', '--img2_height', type=int)
    parser.add_argument('-w1', '--img1_width', type=int)
    parser.add_argument('-w2', '--img2_width', type=int)
    parser.add_argument('--fps1', type=float)
    parser.add_argument('--fps2', type=float)
    parser.add_argument("--mixamo1", help='if v1 belongs to mixamo', action="store_true")
    parser.add_argument("--mixamo2", help='if v2 belongs to mixamo', action="store_true")
    parser.add_argument('--color1', type=str, default='#ff0000##aa0000#550000', help='color1')
    parser.add_argument('--color2', type=str, default='#0000ff#0000aa#000055', help='color2')
    parser.add_argument('--smooth', action='store_true',
                        help="to smooth the output using gaussian kernel")
    parser.add_argument('--transparency', action='store_true',
                        help="make background transparent in resulting frames")
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

    since = time.time()

    input1 = data1['input'].to(device)
    input2 = data2['input'].to(device)
    vlen = min(input1.shape[-1], input2.shape[-1], args.max_length) // 16 * 16
    input1 = input1[:, :, :vlen]
    input2 = input2[:, :, :vlen]

    out12 = net.transfer(input1, input2)
    out21 = net.transfer(input2, input1)

    input1 = input1.detach().cpu().numpy()[0].reshape(-1, 2, input1.shape[-1])
    input2 = input2.detach().cpu().numpy()[0].reshape(-1, 2, input2.shape[-1])
    out12 = out12.detach().cpu().numpy()[0].reshape(-1, 2, out12.shape[-1])
    out21 = out21.detach().cpu().numpy()[0].reshape(-1, 2, out21.shape[-1])

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

    if args.smooth:
        out12 = gaussian_filter1d(out12, sigma=2, axis=-1)
        out21 = gaussian_filter1d(out21, sigma=2, axis=-1)

    total_time = time.time() - since

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
        motion2video(input1, h1, w1, save_dir, 'input1', args.color1, args.transparency, fps=args.fps1)
        motion2video(input2, h2, w2, save_dir, 'input2', args.color2, args.transparency, fps=args.fps2)
        motion2video(out12, h2, w2, save_dir, 'output12', args.color2, args.transparency, fps=args.fps1)
        motion2video(out21, h1, w1, save_dir, 'output21', args.color1, args.transparency, fps=args.fps2)
