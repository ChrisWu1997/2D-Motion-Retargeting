import os
import warnings
from common import config
import numpy as np
import torch
import argparse
import json
from tqdm import tqdm
warnings.filterwarnings("ignore")
from visulization import joints2image, hex2rgb, interpolate_color
import cv2
import imageio
from dataset import MEAN_POSE, STD_POSE
from motion import *
from scipy.ndimage import gaussian_filter1d
import json
from utils import ensure_dir


# used for openpose format
def get_data(json_dir, is_mixamo, is_our_output, scale=1.0):
    def preprocessing(motion):
        if is_mixamo:
            motion_proj = trans_motion(motion)
        else:
            motion_proj = trans_motion2d(motion)

        motion_proj = normalize_motion(motion_proj, MEAN_POSE, STD_POSE)
        motion_proj = motion_proj.reshape((-1, motion_proj.shape[-1]))  # reshape to (joints*2, len_frames)

        motion_proj = torch.Tensor(motion_proj).unsqueeze(0)

        return motion_proj

    motion = json2motion(json_dir, scale, is_mixamo=is_mixamo, is_our_output=is_our_output)
    #motion = motion * scale
    motion_input = preprocessing(motion)

    data = {'input': motion_input, 'raw': motion}
    return data


def vec_interpolate(v1, v2, alphas, repeat_row=0, repeat_col=0):
    if repeat_row == repeat_col == 0:
        return torch.cat([(1 - alpha) * v1 + alpha * v2 for alpha in alphas], dim=0)
    elif repeat_row > 0:
        assert repeat_col == 0
        return torch.cat([(1 - alpha) * v1 + alpha * v2 for alpha in alphas], dim=0).repeat(repeat_row, 1, 1)
    elif repeat_col > 0:
        assert repeat_row == 0
        return torch.cat([((1 - alpha) * v1 + alpha * v2).repeat(repeat_col, 1, 1) for alpha in alphas], dim=0)
    else:
        raise ValueError


def interpolate(net, nr_sample, mode, form, device):
    m1 = net.mot_encoder(input1)
    m2 = net.mot_encoder(input2)
    b1 = net.body_encoder(input1[:, :-2, :])
    b2 = net.body_encoder(input2[:, :-2, :])
    v1 = net.view_encoder(input1[:, :-2, :])
    v2 = net.view_encoder(input2[:, :-2, :])

    alphas = torch.linspace(0, 1, nr_sample).to(device)

    def interpolate_as_form(a1, a2, b1, b2, c1):
        if form == 'line':
            a_mix = vec_interpolate(a1, a2, alphas)
            b_mix = vec_interpolate(b1, b2, alphas)
            c1 = c1.repeat(nr_sample, 1, 1)
        elif form == 'matrix':
            a_mix = vec_interpolate(a1, a2, alphas, repeat_col=nr_sample)
            b_mix = vec_interpolate(b1, b2, alphas, repeat_row=nr_sample)
            c1 = c1.repeat(nr_sample * nr_sample, 1, 1)
        else:
            raise NameError
        return a_mix, b_mix, c1

    if mode == 'motion':
        b_mix, v_mix, m1 = interpolate_as_form(b1, b2, v1, v2, m1)
        dec_input = torch.cat([m1, b_mix.repeat(1, 1, m1.shape[-1]), v_mix.repeat(1, 1, m1.shape[-1])], dim=1)
        out12 = net.decoder(dec_input)

    elif mode == 'body':
        m_mix, v_mix, b1 = interpolate_as_form(m1, m2, v1, v2, b1)
        dec_input = torch.cat([m_mix, b1.repeat(1, 1, m1.shape[-1]), v_mix.repeat(1, 1, m1.shape[-1])], dim=1)
        out12 = net.decoder(dec_input)

    elif mode == 'view':
        m_mix, b_mix, v1 = interpolate_as_form(m1, m2, b1, b2, v1)
        dec_input = torch.cat([m_mix, b_mix.repeat(1, 1, m1.shape[-1]), v1.repeat(1, 1, m1.shape[-1])], dim=1)
        out12 = net.decoder(dec_input)

    elif mode == 'none':
        assert form == 'line'
        m_mix = vec_interpolate(m1, m2, alphas)
        b_mix = vec_interpolate(b1, b2, alphas)
        v_mix = vec_interpolate(v1, v2, alphas)
        dec_input = torch.cat([m_mix, b_mix.repeat(1, 1, m1.shape[-1]), v_mix.repeat(1, 1, m1.shape[-1])], dim=1)
        out12 = net.decoder(dec_input)

    else:
        raise NameError

    return out12


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
    parser.add_argument('--use_tgt_vel', action='store_true',
                        help="to use the second input's velocity for output results if set")
    parser.add_argument('-o', '--out_path', type=str,
                        help='filepath to write the output video, no need to specify this if write_json is actiavted')
    parser.add_argument('--write_json', type=str,
                        help='the folder to write output json')
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False)
    parser.add_argument("--mixamo1", help='if v1 belongs to mixamo', action="store_true")
    parser.add_argument("--mixamo2", help='if v2 belongs to mixamo', action="store_true")
    parser.add_argument("--our_output1", help='if v1 is an output of our system', action="store_true")
    parser.add_argument("--our_output2", help='if v2 is an output of our system', action="store_true")
    args = parser.parse_args()

    if args.mode == 'none':
        assert args.form == 'line'

    if args.write_json is not None:
        ensure_dir(args.write_json)

    # save the arguments for later reproduce
    if args.write_json:
        param_path = os.path.join(args.write_json, 'args.json')
    elif args.out_path:
        param_path = args.out_path[:-3] + 'args.json'
    else:
        param_path = './args.json'
    with open(param_path, 'w') as fp:
        json.dump(args.__dict__, fp, indent=4, sort_keys=True)

    # clip and pad the video
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

    # set network
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = torch.load(args.network)['net'].to(device)
    net.eval()

    # process input data
    data1 = get_data(args.vid1_json_dir, args.mixamo1, args.our_output1, scale1)
    data2 = get_data(args.vid2_json_dir, args.mixamo2, args.our_output2, scale2)

    input1 = data1['input'].to(device)
    input2 = data2['input'].to(device)
    vlen = min(input1.shape[-1], input2.shape[-1], args.max_length) // 16 * 16
    input1 = input1[:, :, :vlen]
    input2 = input2[:, :, :vlen]

    # interpolation
    nr_sample = vlen if args.nr_sample == -1 else args.nr_sample
    print('nr_samples:', args.nr_sample, 'mode:', args.mode, 'form:', args.form)

    out12 = interpolate(net, nr_sample, args.mode, args.form, device)

    input1 = input1.detach().cpu().numpy()[0].reshape(-1, 2, input1.shape[-1])
    input2 = input2.detach().cpu().numpy()[0].reshape(-1, 2, input2.shape[-1])
    out12 = out12.detach().cpu().numpy()#.reshape(-1, 2, out12.shape[-1])

    input1 = normalize_motion_inv(input1, MEAN_POSE, STD_POSE)
    input2 = normalize_motion_inv(input2, MEAN_POSE, STD_POSE)
    # interpolated motions [(J, 2, L), ..., (J, 2, L)]
    if args.use_tgt_vel:
        interp_motions = [trans_motion_inv(normalize_motion_inv(out12[i, :, :], MEAN_POSE, STD_POSE),
                                           velocity=input2[-1])
                          for i in range(out12.shape[0])]
    else:
        interp_motions = [trans_motion_inv(normalize_motion_inv(out12[i, :, :], MEAN_POSE, STD_POSE))
                          for i in range(out12.shape[0])]

    # uncomment this to get back input1 and input2
    #input1 = trans_motion_inv(normalize_motion_inv(input1, MEAN_POSE, STD_POSE))
    #input2 = trans_motion_inv(normalize_motion_inv(input2, MEAN_POSE, STD_POSE))


    # each cell's position
    if args.form == 'line':
        position = [str(i) for i in range(len(interp_motions))]
    else:
        position = [str(i // nr_sample) + '.' + str(i % nr_sample) for i in range(len(interp_motions))]

    # write json
    if args.write_json is not None:
        ensure_dir(args.write_json)
        out1_dir_json = os.path.join(args.write_json, 'json/point-input1')
        out2_dir_json = os.path.join(args.write_json, 'json/point-input2')
        out1_dir_vid = os.path.join(args.write_json, 'frames/point-input2')
        out2_dir_vid = os.path.join(args.write_json, 'frames/point-input2')
        ensure_dir(out1_dir_json)
        ensure_dir(out2_dir_json)
        ensure_dir(out1_dir_vid)
        ensure_dir(out2_dir_vid)
        motion2json(input1, h1, w1, out1_dir_json)
        motion2json(input2, h2, w2, out2_dir_json)

        for i, motion in enumerate(interp_motions):
            out_dir_json = os.path.join(args.write_json, 'json/point-{}'.format(position[i]))
            out_dir_frames = os.path.join(args.write_json, 'frames/point-{}'.format(position[i]))
            ensure_dir(out_dir)
            ensure_dir(out_dir_frames)
            motion2json(motion, h2, w2, out_dir_json)
            color = interpolate_color(color1_rgb, color2_rgb, alpha)
            for i in tqdm(range(vlen)):
                img = joints2image(motion[:, :, i], color, H=h2, W=w2)
        print('Json files are writen.')

    # write video

    if out_path is not None:
        print('generating video...')
        cell_height = args.cell_height
        cell_width = pad_to_16x(int(w2 * cell_height / h2))
        vlen = min(input1.shape[-1], input2.shape[-1])
        color1 = hex2rgb(args.color1)
        color2 = hex2rgb(args.color2)

        for j, motion in enumerate(interp_motions):
            out_path = os.path.join(args.write_json, 'interpolation.mp4') if args.write_json is not None else args.out_path
            videowriter = imageio.get_writer(out_path, fps=25)
            color = interpolate_color(color1, color2, j/len(interp_motions))
            for i in tqdm(range(vlen)):
            img_iterps = []
                img = joints2image(motion[:, :, i], color, H=h2, W=w2)

                cv2.putText(img, position[j], (40, 40), 1, 3, (0, 0, 255), 1)
                img = cv2.resize(img, (cell_width, cell_height))
                img_iterps.append(img)

            if args.form == 'line':
                whole_img = np.concatenate(img_iterps, axis=1)
            else:
                img_rows = [np.concatenate(img_iterps[j * nr_sample: (j + 1) * nr_sample], axis=1)
                            for j in range(nr_sample)]
                whole_img = np.concatenate(img_rows, axis=0)
            videowriter.append_data(whole_img)
        videowriter.close()
        print('Video is written.')

        # for i in tqdm(range(vlen)):
        #     img_iterps = []
        #     for j, motion in enumerate(interp_motions):
        #         img = joints2image(motion[:, :, i], H=h2, W=w2)
        #
        #         cv2.putText(img, position[j], (40, 40), 1, 3, (0, 0, 255), 1)
        #         img = cv2.resize(img, (cell_width, cell_height))
        #         img_iterps.append(img)
        #
        #     if args.form == 'line':
        #         whole_img = np.concatenate(img_iterps, axis=1)
        #     else:
        #         img_rows = [np.concatenate(img_iterps[j * nr_sample: (j + 1) * nr_sample], axis=1)
        #                     for j in range(nr_sample)]
        #         whole_img = np.concatenate(img_rows, axis=0)
        #     videowriter.append_data(whole_img)
        # videowriter.close()
        # print('Video is written.')
