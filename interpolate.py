from scipy.ndimage import gaussian_filter1d
import torch
import numpy as np
import argparse
from tqdm import tqdm
import os
import cv2
import imageio
from dataset import get_meanpose
from model import get_autoencoder
from functional.visualization import hex2rgb, joints2image, interpolate_color
from functional.motion import preprocess_motion2d, postprocess_motion2d, openpose2motion
from functional.utils import ensure_dir, pad_to_height
from common import config


def vec_interpolate(v1, v2, alphas, repeat_row=0, repeat_col=0):
    """interpolate two vectors"""
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
    """interpolate between network latent space"""
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="filepath for trained model weights")
    parser.add_argument('-v1', '--vid1_json_dir', type=str, help="video1's openpose json directory")
    parser.add_argument('-v2', '--vid2_json_dir', type=str, help="video2's openpose json directory")
    parser.add_argument('-h1', '--img1_height', type=int, help="video1's height")
    parser.add_argument('-w1', '--img1_width', type=int, help="video1's width")
    parser.add_argument('-h2', '--img2_height', type=int, help="video2's height")
    parser.add_argument('-w2', '--img2_width', type=int, help="video2's width")
    parser.add_argument('-o', '--out_path', type=str, help='filepath to write the output video')
    parser.add_argument('--keep_attr', type=str, choices=['motion', 'body', 'view', 'none'], default='none',
                        help='which attribute to keep')
    parser.add_argument('--form', type=str, choices=['matrix', 'line'], default='line', help='which form of output')
    parser.add_argument('--nr_sample', type=int, default=5, help='how many samples to interpolate')
    parser.add_argument('--color1', type=str, default='#ff0000##aa0000#550000', help='color1')
    parser.add_argument('--color2', type=str, default='#0000ff#0000aa#000055', help='color2')
    parser.add_argument('-ch', '--cell_height', type=int, default=128, help="cell's height when saving the video")
    parser.add_argument('--max_length', type=int, default=120, help='maximum input video length')
    parser.add_argument('--transparency', action='store_true', help="make background transparent in resulting frames")
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False)
    args = parser.parse_args()

    config.initialize(args)

    # if keep no attribute, interpolate over all three latent space
    if args.keep_attr == 'none':
        assert args.form == 'line'

    # clip and pad the video
    h1, w1, scale1 = pad_to_height(config.img_size[0], args.img1_height, args.img1_width)
    h2, w2, scale2 = pad_to_height(config.img_size[0], args.img2_height, args.img2_width)

    # load trained model
    net = get_autoencoder(config)
    net.load_state_dict(torch.load(args.model_path))
    net.to(config.device)
    net.eval()

    # mean/std pose
    mean_pose, std_pose = get_meanpose(config)

    # process input data
    input1 = openpose2motion(args.vid1_json_dir, scale=scale1, max_frame=args.max_length)
    input2 = openpose2motion(args.vid2_json_dir, scale=scale2, max_frame=args.max_length)
    if input1.shape[-1] != input2.shape[-1]:
        length = min(input1.shape[-1], input2.shape[-1])
        input1 = input1[:, :, length]
        input2 = input2[:, :, length]
    input1 = preprocess_motion2d(input1, mean_pose, std_pose)
    input2 = preprocess_motion2d(input2, mean_pose, std_pose)
    input1 = input1.to(config.device)
    input2 = input2.to(config.device)

    # interpolation
    print('nr_samples:', args.nr_sample, 'mode:', args.keep_attr, 'form:', args.form)

    out12 = interpolate(net, args.nr_sample, args.keep_attr, args.form, config.device)

    input1 = postprocess_motion2d(input1, mean_pose, std_pose, w1 // 2, h1 // 2)
    input2 = postprocess_motion2d(input2, mean_pose, std_pose, w2 // 2, h2 // 2)

    # interpolated motions [(J, 2, L), ..., (J, 2, L)]
    interp_motions = [postprocess_motion2d(out12[i:i+1, :, :], mean_pose, std_pose) for i in range(out12.shape[0])]

    # each cell's position
    if args.form == 'line':
        position = [str(i) for i in range(len(interp_motions))]
    else:
        position = [str(i // args.nr_sample) + '.' + str(i % args.nr_sample) for i in range(len(interp_motions))]

    # write output video
    out_path = args.out_path
    if out_path is not None:
        pardir = os.path.split(out_path)[0]
        ensure_dir(pardir)
        print('generating video...')
        cell_height = cell_width = args.cell_height
        color1 = hex2rgb(args.color1)
        color2 = hex2rgb(args.color2)
        vlen = min(input1.shape[-1], input2.shape[-1])

        videowriter = imageio.get_writer(out_path, fps=25)
        for i in tqdm(range(vlen)):
            img_iterps = []
            for j, motion in enumerate(interp_motions):
                if args.form == 'line':
                    color = interpolate_color(color1, color2, j / (args.nr_sample - 1))
                else:
                    color = interpolate_color(color1, color2, (j // args.nr_sample) / (args.nr_sample - 1))
                img, img_cropped = joints2image(motion[:, :, i], color, transparency=args.transparency,
                                                H=config.img_size[0], W=config.img_size[0])
                img = cv2.resize(img, (cell_width, cell_height))
                img_iterps.append(img)

            if args.form == 'line':
                whole_img = np.concatenate(img_iterps, axis=1)
            else:
                img_rows = [np.concatenate(img_iterps[j * args.nr_sample: (j + 1) * args.nr_sample], axis=1)
                            for j in range(args.nr_sample)]
                whole_img = np.concatenate(img_rows, axis=0)

            videowriter.append_data(whole_img)
        videowriter.close()
        print('Video is written.')
