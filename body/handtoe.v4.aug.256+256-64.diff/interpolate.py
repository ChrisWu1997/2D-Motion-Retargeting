import os
from common import config
import numpy as np
import torch
import argparse
import json
import torch.nn.functional as F
from tqdm import tqdm
from visualization import pose2im_all, joints2image, hex2rgb, interpolate_color
import cv2
import imageio
from dataset import MEAN_POSE, STD_POSE
from motion import normalize_motion, normalize_motion_inv, trans_motion_inv, trans_motion2d, json2motion, trans_motion
from scipy.ndimage import gaussian_filter1d
import time
from utils import ensure_dir, motion2openpose, save_image

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

def motion2video(motion, h, w, save_path, colors, transparency=False, motion_tgt=None, fps=25):
    videowriter = imageio.get_writer(os.path.join(save_path, 'video.mp4'), fps=fps)
    vlen = motion.shape[-1]
    frames_dir = os.path.join(save_path)
    ensure_dir(frames_dir)
    for i in tqdm(range(vlen)):
        print("motion[:, :, i]: ", motion[:, :, i])
        [img, img_cropped] = joints2image(motion[:, :, i], colors, transparency, H=h, W=w)
        if motion_tgt is not None:
            [img_tgt, img_tgt_cropped] = joints2image(motion_tgt[:, :, i], colors, H=h, W=w)
            img = cv2.addWeighted(img_tgt, 0.3, img, 0.7, 0)
        save_image(img_cropped, os.path.join(frames_dir, "%04d.png" % i))
        videowriter.append_data(img)
    videowriter.close()
# def get_data(json_dir, is_mixamo, scale=1.0):
#     paths = sorted(os.listdir(json_dir))
#     paths = [os.path.join(json_dir, x) for x in paths]
#
#     def json2motion(json_list):
#         mot = []
#         for filename in json_list:
#             with open(filename) as f:
#                 jointDict = json.load(f)
#                 if is_mixamo:
#                     pose_joints = np.array(jointDict['pose_keypoints_3d']).reshape((-1, 3))[:15]
#                     face_joints = np.array(jointDict['face_keypoints_3d']).reshape((-1, 3))
#                     joint = np.r_[pose_joints, face_joints]
#                 else:
#                     joint = np.array(jointDict['people'][0]['pose_keypoints_2d']).reshape((-1, 3))[:17, :2]
#                 if len(mot) > 0:
#                     joint[np.where(joint == 0)] = mot[-1][np.where(joint == 0)]
#                 mot.append(joint)
#
#         mot = np.stack(mot, axis=2)
#         mot = gaussian_filter1d(mot, sigma=2, axis=-1)
#         if is_mixamo:
#             mot = mot[:, [0, 2], :]  # (15, 2, 64)
#             mot[:, 1, :] = - mot[:, 1, :]
#
#             mot = mot * config.unit
#
#             mot[1, :, :] = (mot[2, :, :] + mot[5, :, :]) / 2
#             mot[8, :, :] = (mot[9, :, :] + mot[12, :, :]) / 2
#
#             centers = mot[8, :, :]
#             mot = mot - centers
#             centers_reset = centers - centers[:, 0].reshape(2, 1) \
#                             + np.array([[config.img_size[0] // 2], [config.img_size[1] // 2]])
#             mot = mot + centers_reset
#         return mot
#
#     def preprocessing(motion):
#         motion_proj = trans_motion2d(motion)
#
#         motion_proj = normalize_motion(motion_proj, MEAN_POSE, STD_POSE)
#         motion_proj = motion_proj.reshape((-1, motion_proj.shape[-1]))  # reshape to (joints*2, len_frames)
#
#         motion_proj = torch.Tensor(motion_proj).unsqueeze(0)
#
#         return motion_proj

    # motion = json2motion(paths)
    # motion = motion * scale
    # motion_input = preprocessing(motion)
    #
    # data = {'input': motion_input, 'raw': motion}
    # return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network', type=str)
    parser.add_argument('-v1', '--vid1_json_dir', type=str)
    parser.add_argument('-v2', '--vid2_json_dir', type=str)
    parser.add_argument('-h1', '--img1_height', type=int)
    parser.add_argument('-h2', '--img2_height', type=int)
    parser.add_argument('-w1', '--img1_width', type=int)
    parser.add_argument('-w2', '--img2_width', type=int)
    parser.add_argument('--fps1', type=float, default=25)
    parser.add_argument('--fps2', type=float, default=25)
    parser.add_argument('--color1', type=str, default='#ff0000##aa0000#550000', help='color1')
    parser.add_argument('--color2', type=str, default='#0000ff#0000aa#000055', help='color2')
    parser.add_argument('-ch', '--cell_height', type=int, default=128,
                        help="cell's height when saving the video")
    parser.add_argument('--max_length', type=int, default=240,
                        help='maximum input video length')
    parser.add_argument('--mode', type=str, choices=['motion', 'body', 'view', 'none'], default='none',
                        help='which attribute to keep')
    parser.add_argument('--form', type=str, choices=['matrix', 'line'], default='line',
                        help='which form of output')
    parser.add_argument('--nr_sample', type=int, default=8,
                        help='how many samples to interpolate')
    parser.add_argument('--transparency', action='store_true',
                        help="make background transparent in resulting frames")
    parser.add_argument('--use_tgt_vel', action='store_true',
                        help="to use the second input's velocity for output results if set")
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
    # out12 = trans_motion_inv(normalize_motion_inv(out12, MEAN_POSE, STD_POSE), w2 // 2, h2 // 2)
    # out21 = trans_motion_inv(normalize_motion_inv(out21, MEAN_POSE, STD_POSE), w1 // 2, h1 // 2)

    total_time = time.time() - since
    print("total time:", total_time)
    print("1/fps:", total_time / max(input1.shape[-1], input2.shape[-1]))


    if args.use_tgt_vel:
        interp_motions = [trans_motion_inv(normalize_motion_inv(out12[i, :, :], MEAN_POSE, STD_POSE),
                                           velocity=input2[-1])
                          for i in range(out12.shape[0])]
    else:
        interp_motions = [trans_motion_inv(normalize_motion_inv(out12[i, :, :], MEAN_POSE, STD_POSE))
                          for i in range(out12.shape[0])]

    def motion2json(motion, h, w, save_dir):
        nr_frames = motion.shape[-1]
        for i in range(nr_frames):
            path = os.path.join(save_dir, "%04d_keypoints.json" % i)
            out_dict = {"resolution": (h, w), 'pose_keypoints': motion[:, :, i].reshape(-1).tolist()}
            with open(path, 'w') as f:
                json.dump(out_dict, f)

    # each cell's position
    if args.form == 'line':
        position = [str(i) for i in range(len(interp_motions))]
    else:
        position = [str(i // nr_sample) + '.' + str(i % nr_sample) for i in range(len(interp_motions))]

    # write json
    if args.write_json is not None:
        ensure_dir(args.write_json)
        out1_dir_json = os.path.join(args.write_json, 'point-input1','point-original')
        out2_dir_json = os.path.join(args.write_json, 'point-input2','point-original')
        out1_dir_vid = os.path.join(args.write_json, 'point-input1','skeleton-original')
        out2_dir_vid = os.path.join(args.write_json, 'point-input2','skeleton-original')
        ensure_dir(out1_dir_json)
        ensure_dir(out2_dir_json)
        ensure_dir(out1_dir_vid)
        ensure_dir(out2_dir_vid)
        motion2openpose(input1, h1, w1, out1_dir_json)
        motion2openpose(input2, h2, w2, out2_dir_json)
        color1 = hex2rgb(args.color1)
        color2 = hex2rgb(args.color1) if args.mode is 'body' else hex2rgb(args.color2)
        # motion2video(input1, h1, w1, out1_dir_vid, color1, args.transparency, fps=args.fps1)
        # motion2video(input2, h2, w2, out2_dir_vid, color2, args.transparency, fps=args.fps2)

        for i, motion in enumerate(interp_motions):
            out_dir_json = os.path.join(args.write_json, 'point-{}'.format(position[i]), 'point-original')
            out_dir_frames = os.path.join(args.write_json, 'point-{}'.format(position[i]), 'skeleton-original')
            ensure_dir(out_dir_json)
            ensure_dir(out_dir_frames)
            motion2json(motion, h2, w2, out_dir_json)
            color = interpolate_color(color2, color1, (i+1)/(len(interp_motions)+1))
            motion2video(motion, h2, w2, out_dir_frames, color, args.transparency, fps=args.fps2)
        print('Json files are writen.')
