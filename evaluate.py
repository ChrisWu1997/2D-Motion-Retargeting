import os
from scipy.ndimage import gaussian_filter1d
import torch
import argparse
import numpy as np
from dataset import get_meanpose, get_dataloader
from model import get_autoencoder
from functional.visualization import motion2video, hex2rgb
from functional.motion import preprocess_motion2d, postprocess_motion2d, openpose2motion
from functional.utils import ensure_dir, pad_to_height
from common import config

VIEW_ANGLES = [(0, 0, -np.pi / 2),
               (0, 0, -np.pi / 3),
               (0, 0, -np.pi / 6),
               (0, 0, 0),
               (0, 0, np.pi / 6),
               (0, 0, np.pi / 3),
               (0, 0, np.pi / 2)]

def handle2x(config, args):
    w1 = h1 = w2 = h2 = 512

    # load trained model
    net = get_autoencoder(config)
    net.load_state_dict(torch.load(args.model_path))
    net.to(config.device)
    net.eval()

    # mean/std pose
    mean_pose, std_pose = get_meanpose(config)

    # get input
    dataloder = get_dataloader('test', config)
    v1 = VIEW_ANGLES[args.view1] if args.view1 is not None else None
    v2 = VIEW_ANGLES[args.view2] if args.view2 is not None else None
    input1 = dataloder.dataset.preprocessing(args.path1, v1).unsqueeze(0)
    input2 = dataloder.dataset.preprocessing(args.path2, v2).unsqueeze(0)
    input1 = input1.to(config.device)
    input2 = input2.to(config.device)

    # transfer by network
    out12 = net.transfer(input1, input2)
    out21 = net.transfer(input2, input1)

    # postprocessing the outputs
    input1 = postprocess_motion2d(input1, mean_pose, std_pose, w1 // 2, h1 // 2)
    input2 = postprocess_motion2d(input2, mean_pose, std_pose, w2 // 2, h2 // 2)
    out12 = postprocess_motion2d(out12, mean_pose, std_pose, w2 // 2, h2 // 2)
    out21 = postprocess_motion2d(out21, mean_pose, std_pose, w1 // 2, h1 // 2)

    if not args.disable_smooth:
        out12 = gaussian_filter1d(out12, sigma=2, axis=-1)
        out21 = gaussian_filter1d(out21, sigma=2, axis=-1)

    if args.out_dir is not None:
        save_dir = args.out_dir
        ensure_dir(save_dir)
        color1 = hex2rgb(args.color1)
        color2 = hex2rgb(args.color2)
        np.savez(os.path.join(save_dir, 'results.npz'),
                 input1=input1,
                 input2=input2,
                 out12=out12,
                 out21=out21)
        if args.render_video:
            print("Generating videos...")
            motion2video(input1, h1, w1, os.path.join(save_dir, 'input1.mp4'), color1, args.transparency,
                         fps=args.fps, save_frame=args.save_frame)
            motion2video(input2, h2, w2, os.path.join(save_dir,'input2.mp4'), color2, args.transparency,
                         fps=args.fps, save_frame=args.save_frame)
            motion2video(out12, h2, w2, os.path.join(save_dir,'out12.mp4'), color2, args.transparency,
                         fps=args.fps, save_frame=args.save_frame)
            motion2video(out21, h1, w1, os.path.join(save_dir,'out21.mp4'), color1, args.transparency,
                         fps=args.fps, save_frame=args.save_frame)
    print("Done.")


def handle3x(config, args):
    # resize input
    h1, w1, scale1 = pad_to_height(config.img_size[0], args.img1_height, args.img1_width)
    h2, w2, scale2 = pad_to_height(config.img_size[0], args.img2_height, args.img2_width)
    h3, w3, scale3 = pad_to_height(config.img_size[0], args.img2_height, args.img3_width)

    # load trained model
    net = get_autoencoder(config)
    net.load_state_dict(torch.load(args.model_path))
    net.to(config.device)
    net.eval()

    # mean/std pose
    mean_pose, std_pose = get_meanpose(config)

    # get input
    input1 = openpose2motion(args.vid1_json_dir, scale=scale1, max_frame=args.max_length)
    input2 = openpose2motion(args.vid2_json_dir, scale=scale2, max_frame=args.max_length)
    input3 = openpose2motion(args.vid3_json_dir, scale=scale3, max_frame=args.max_length)
    input1 = preprocess_motion2d(input1, mean_pose, std_pose)
    input2 = preprocess_motion2d(input2, mean_pose, std_pose)
    input3 = preprocess_motion2d(input3, mean_pose, std_pose)
    input1 = input1.to(config.device)
    input2 = input2.to(config.device)
    input3 = input3.to(config.device)

    # transfer by network
    out = net.transfer_three(input1, input2, input3)

    # postprocessing the outputs
    input1 = postprocess_motion2d(input1, mean_pose, std_pose, w1 // 2, h1 // 2)
    input2 = postprocess_motion2d(input2, mean_pose, std_pose, w2 // 2, h2 // 2)
    input3 = postprocess_motion2d(input3, mean_pose, std_pose, w2 // 2, h2 // 2)
    out = postprocess_motion2d(out, mean_pose, std_pose, w2 // 2, h2 // 2)

    if not args.disable_smooth:
        out = gaussian_filter1d(out, sigma=2, axis=-1)

    if args.out_dir is not None:
        save_dir = args.out_dir
        ensure_dir(save_dir)
        color1 = hex2rgb(args.color1)
        color2 = hex2rgb(args.color2)
        color3 = hex2rgb(args.color3)
        np.savez(os.path.join(save_dir, 'results.npz'),
                 input1=input1,
                 input2=input2,
                 input3=input3,
                 out=out)
        if args.render_video:
            print("Generating videos...")
            motion2video(input1, h1, w1, os.path.join(save_dir,'input1.mp4'), color1, args.transparency,
                         fps=args.fps, save_frame=args.save_frame)
            motion2video(input2, h2, w2, os.path.join(save_dir,'input2.mp4'), color2, args.transparency,
                         fps=args.fps, save_frame=args.save_frame)
            motion2video(input3, h3, w3, os.path.join(save_dir,'input3.mp4'), color3, args.transparency,
                         fps=args.fps, save_frame=args.save_frame)
            motion2video(out, h2, w2, os.path.join(save_dir,'out.mp4'), color2, args.transparency,
                         fps=args.fps, save_frame=args.save_frame)

    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, choices=['skeleton', 'view', 'full'], required=True,
                        help='which structure to use.')
    parser.add_argument('--model_path', type=str, required=True, help="filepath for trained model weights")
    parser.add_argument('--path1', type=str)
    parser.add_argument('--path2', type=str)
    parser.add_argument('--view1', type=int)
    parser.add_argument('--view2', type=int)
    parser.add_argument('-o', '--out_dir', type=str, default='./outputs', help="output saving directory")
    parser.add_argument('--render_video', type=bool, default=True, help="whether to save rendered video")
    parser.add_argument('--fps', type=float, default=25, help="fps of output video")
    parser.add_argument('--save_frame', action='store_true', help="to save rendered video frames")
    parser.add_argument('--color1', type=str, default='#a50b69#b73b87#db9dc3', help='color1')
    parser.add_argument('--color2', type=str, default='#4076e0#40a7e0#40d7e0', help='color2')
    parser.add_argument('--color3', type=str, default='#ff8b06#ffb431#ffcd9d', help='color3')
    parser.add_argument('--disable_smooth', action='store_true',
                        help="disable gaussian kernel smoothing")
    parser.add_argument('--transparency', action='store_true',
                        help="make background transparent in resulting frames")
    parser.add_argument('--max_length', type=int, default=120,
                        help='maximum input video length')
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False)
    args = parser.parse_args()

    config.initialize(args)

    if args.name == 'full':
        handle3x(config, args)
    else:
        handle2x(config, args)


if __name__ == '__main__':
    main()

