import os
from common import config
import numpy as np
import torch
import argparse
import json
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import cv2
import imageio
from dataset import MEAN_POSE, STD_POSE
from motion import trans_motion2d, normalize_motion, json2motion
from scipy.ndimage import gaussian_filter1d
from utils import ensure_dir
matplotlib.use('Agg')


def get_data(json_dir, is_mixamo, scale=1.0, person=0, start=0, end=-1):
    paths = sorted(os.listdir(json_dir))[start:end]
    paths = [os.path.join(json_dir, x) for x in paths]

    def preprocessing(motion):
        motion = motion * scale
        motion_proj, _ = trans_motion2d(motion)

        motion_proj = normalize_motion(motion_proj, MEAN_POSE, STD_POSE)
        motion_proj = motion_proj.reshape((-1, motion_proj.shape[-1]))  # reshape to (joints*2, len_frames)

        motion_proj = torch.Tensor(motion_proj).unsqueeze(0)

        return motion_proj

    motion = json2motion(paths, is_mixamo=is_mixamo, person=person)
    motion_input = preprocessing(motion)

    data = {'input': motion_input, 'raw': motion}
    return data


def calc_receptive_field(k, s, p, nr_layer):

    def forward(j_in, r_in, start_in):
        j_out = j_in * s
        r_out = r_in + (k - 1) * j_in
        start_out = start_in + ((k - 1) / 2 - p) * j_in
        return j_out, r_out, start_out

    j = 1
    r = 1
    start = 0.5
    for i in range(nr_layer):
        j, r, start = forward(j, r, start)
    return j, r, start


def inverse_mapping(left, right, kernel_size=8, stride=2, padding=3, nr_layer=3):
    """
        map an interval in latent space to
        its receptive field in original input space
    """

    # calculate the receptive field for the latent space
    def forward(j_in, r_in, start_in):
        j_out = j_in * stride
        r_out = r_in + (kernel_size - 1) * j_in
        start_out = start_in + ((kernel_size - 1) / 2 - padding) * j_in
        return j_out, r_out, start_out

    j = 1
    r = 1
    start = 0.5
    for i in range(nr_layer):
        j, r, start = forward(j, r, start)

    # corresponding interval in input(layer0)
    ll = int(start + left * j - r / 2)
    rr = int(start + right * j + r / 2)

    return ll, rr


def measure_similarity(m1mat, m2mat, top_n=5):
    m1mat_norm = m1mat / np.linalg.norm(m1mat, axis=0, keepdims=True)
    m2mat_norm = m2mat / np.linalg.norm(m2mat, axis=0, keepdims=True)
    window_size = m2mat.shape[-1]
    if m1mat.shape[-1] < window_size:
        return 0, [0, 0]
    scores = []

    for i in range(m1mat.shape[-1] - window_size + 1):
        m1vec = m1mat_norm[:, i:i + window_size]
        score = np.sum(m1vec * m2mat_norm) / window_size
        scores.append(score)

    idx_topn = np.argsort(scores)[-top_n:][::-1]
    scores_topn = [scores[idx] for idx in idx_topn]
    intervals_topn = [[idx, idx + window_size - 1] for idx in idx_topn]

    return scores, intervals_topn, scores_topn


def clip_and_save_video(vid_path, out_path, left: int, right: int):
    reader = imageio.get_reader(vid_path)
    fps = reader.get_meta_data()['fps']

    writer = imageio.get_writer(out_path, fps=fps)
    for i in tqdm(range(left, right + 1, 1)):
        frame = reader.get_data(i)
        writer.append_data(frame)
    reader.close()
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network', type=str)
    parser.add_argument('-v1', '--vid1_json_dir', help='long motion sequence', type=str)
    parser.add_argument('-v2', '--vid2_json_dir', help='query motion', type=str)
    parser.add_argument('--vid_path', help='base video path, for result clipping', type=str)
    parser.add_argument('-h1', '--height1', type=int)
    parser.add_argument('-w1', '--width1', type=int)
    parser.add_argument('-h2', '--height2', type=int)
    parser.add_argument('-w2', '--width2', type=int)
    parser.add_argument('-t', '--top_n', type=int, default=5)
    parser.add_argument('--flip', action="store_true")
    parser.add_argument("--mixamo1", help='if query belongs to mixamo', action="store_true")
    parser.add_argument("--mixamo2", help='if query belongs to mixamo', action="store_true")
    parser.add_argument('-o', '--out_dir', default='./query', type=str)
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = torch.load(args.network)['net'].to(device)

    h1, w1 = args.height1, args.width1
    scale1 = config.img_size[0] / h1
    h2, w2 = args.height2, args.width2
    scale2 = config.img_size[0] / h2

    data1 = get_data(args.vid1_json_dir, is_mixamo=args.mixamo1, scale=scale1)
    data2 = get_data(args.vid2_json_dir, is_mixamo=args.mixamo2, scale=scale2)

    input1 = data1['input'].to(device)
    input2 = data2['input'].to(device)

    print('len of input1:', input1.shape[-1])
    print('len of input2:', input2.shape[-1])

    m1mat = net.mot_encoder(input1)
    m2mat = net.mot_encoder(input2)

    m1mat = m1mat.detach().cpu().numpy()[0]
    m2mat = m2mat.detach().cpu().numpy()[0]

    # measure similarity in motion latent space
    scores, interval_topn, scores_topn = measure_similarity(m1mat, m2mat, top_n=args.top_n)
    idx = np.arange(len(scores))
    plt.plot(idx, scores)

    # find corresponding receptive field
    interval_topn = [inverse_mapping(item[0], item[1]) for item in interval_topn]
    interval_topn = [[max(0, item[0]), min(input1.shape[-1], item[1])] for item in interval_topn]
    print('top {} similar intervals:'.format(args.top_n))
    for j, item in enumerate(interval_topn):
        print(' [{}, {}]'.format(item[0], item[1]), 'score: {}'.format(scores_topn[j]))

    if args.out_dir and args.vid_path:
        ensure_dir(args.out_dir)
        plt.savefig(args.out_dir + '/scores.png')
        with open(args.out_dir + '/scores.json', 'w') as f:
            json.dump(scores, f)

        for i, item in enumerate(interval_topn):
            clip_and_save_video(args.vid_path, args.out_dir + '/top_{}.mp4'.format(i), item[0], item[1])
