import os
from common import config
import numpy as np
import torch
import argparse
import json
from tqdm import tqdm
from utils import ensure_dir
import imageio
import time
from dataset import MEAN_POSE, STD_POSE
from motion import trans_motion2d, normalize_motion, json2motion
from scipy.ndimage import gaussian_filter1d
from scipy.io import loadmat


NET_PATH = './model/epoch300.pth.tar'
DATA_ROOT = '/data1/wurundi/Penn_Action/motions'
#VIDEO_DIR = '/data1/wurundi/Penn_Action/videos'


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


class QueryEngine(object):
    def __init__(self, data_root, net_path, device):
        self.data_root = data_root
        #self.video_dir = video_dir

        self.net = torch.load(net_path)['net'].to(device)
        self.net.eval()
        self.device = device

        self.base_dir = './assets/database-penn'
        self.motbase_path = os.path.join(self.base_dir, 'mot-database.npy')
        self.bodybase_path = os.path.join(self.base_dir, 'body-database.npy')
        self.info_path = os.path.join(self.base_dir, 'info.json')
        ensure_dir(self.base_dir)

        if not os.path.exists(self.motbase_path) or not os.path.exists(self.bodybase_path) \
                or not os.path.exists(self.info_path):
            self.build_database()
        self.motbase = np.load(self.motbase_path)
        self.bodybase = np.load(self.bodybase_path)
        with open(self.info_path) as fp:
            self.info = json.load(fp)

        self.nr_items = len(self.info)

    @staticmethod
    def preprocessing(motion2d, scale):
        motion2d = motion2d * scale
        motion_proj, _ = trans_motion2d(motion2d)

        motion_proj = normalize_motion(motion_proj, MEAN_POSE, STD_POSE)
        motion_proj = motion_proj.reshape((-1, motion_proj.shape[-1]))
        motion_proj = torch.Tensor(motion_proj).unsqueeze(0)

        return motion_proj

    def build_database(self):
        print('generating database...')
        since = time.time()

        motmat_base = []
        bodymat_base = []
        info_base = []
        end = 0

        label_dir = self.data_root.replace('motions', 'labels')
        motion_names = sorted(os.listdir(self.data_root))
        for filename in motion_names:
            path = os.path.join(self.data_root, filename)
            motion = np.load(path)

            vid_name = filename.split('_')[0]
            label_path = os.path.join(label_dir, vid_name + '.mat')
            content = loadmat(label_path)
            H, W = content['dimensions'].tolist()[0][:2]

            motion_input = self.preprocessing(motion, config.img_size[0] / H).to(device)
            motmat = self.net.mot_encoder(motion_input)
            bodymat = self.net.body_encoder(motion_input)
            motmat = motmat.detach().cpu().numpy()[0]
            bodymat = bodymat.detach().cpu().numpy()[0]

            motmat_base.append(motmat)
            bodymat_base.append(bodymat)

            info = {'name': filename[:4],
                    #'video_path': os.path.join(self.video_dir, '{}.mp4'.format(vid_name)),
                    'action': content['action'].tolist()[0],
                    'pose': content['pose'].tolist()[0],
                    'nr_frames': motion.shape[-1],
                    'resolution': [H, W],
                    'database_clip': [end, end + motmat.shape[-1]]}
            info_base.append(info)
            end = end + motmat.shape[-1]

        motmat_base = np.concatenate(motmat_base, axis=1)
        bodymat_base = np.concatenate(bodymat_base, axis=1)
        print('motbase shape:', motmat_base.shape)
        print('bodybase shape:', bodymat_base.shape)
        np.save(self.motbase_path, motmat_base)
        np.save(self.bodybase_path, bodymat_base)

        print('Number of items in database:', len(info_base))
        with open(self.info_path, 'w') as f:
            json.dump(info_base, f)

        print('Database is created. Total time: {:3f}'.format(time.time() - since))

    @staticmethod
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

    @staticmethod
    def measure_similarity(m1mat, m2mat):
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

        idx_top = np.argmax(scores)
        score_top = scores[idx_top]
        interval_top = [idx_top, idx_top + window_size - 1]

        return score_top, interval_top

    def query(self, motion_input, require_view=False, top_n=5):
        motmat = self.net.mot_encoder(motion_input.to(self.device))
        motmat = motmat.detach().cpu().numpy()[0]

        all_scores = []
        all_intervals = []
        for i in range(self.nr_items):
            base_motmat = self.motbase[:, self.info[i]['database_clip'][0]:self.info[i]['database_clip'][1]]
            # measure similarity in motion latent space
            score, interval = self.measure_similarity(base_motmat, motmat)
            # mapping to receptive field
            interval_rf = self.inverse_mapping(interval[0], interval[1])
            interval_rf = [max(0, interval_rf[0]), min(self.info[i]['nr_frames'] - 1, interval_rf[1])]

            all_scores.append(score)
            all_intervals.append(interval_rf)

        if not require_view:
            topn_idx = np.argsort(all_scores)[-top_n:][::-1]
            result = [{'info': self.info[idx],
                       'motion score': round(float(all_scores[idx]), 6),
                       'interval': all_intervals[idx],
                        } for idx in topn_idx]

        else:
            # search in the top_n motion result
            # this part is not completed
            bodymat = self.net.body_encoder(motion_input.to(self.device))
            bodymat = bodymat.detach().cpu().numpy()[0]
            mot_top2n_idx = np.argsort(all_scores)[-top_n:][::-1]
            body_scores = []
            for i in mot_top2n_idx:
                base_bodymat = self.bodybase[:, i][:, np.newaxis]
                score, _ = self.measure_similarity(base_bodymat, bodymat)
                body_scores.append(score)
            body_idx = np.argsort(body_scores)[-top_n:][::-1]
            topn_idx = [mot_top2n_idx[i] for i in body_idx]

            result = [{'info': self.info[idx],
                       'motion score': round(float(all_scores[idx]), 6),
                       'view score': round(float(body_scores[mot_top2n_idx.tolist().index(idx)]), 6),
                       'interval': all_intervals[idx],} for idx in topn_idx]
        return result


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
    parser.add_argument('-q', '--query_json_dir', help="query motion's json dir, openpose format", type=str)
    parser.add_argument('--start', help="query motion's start frame", type=int, default=0)
    parser.add_argument('--end', help="query motion's end frame", type=int, default=-1)
    parser.add_argument('-ih', '--height', type=int)
    parser.add_argument('-iw', '--width', type=int)
    parser.add_argument('-p', '--person', type=int, default=0)
    parser.add_argument('--require_view', help='require similar view', action="store_true")
    parser.add_argument("--mixamo", help='if query belongs to mixamo', action="store_true")
    parser.add_argument('-t', '--top_n', type=int, default=5)
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False)
    parser.add_argument('--write_json', help='directory to write result information', type=str)
    parser.add_argument('--vis_dir', type=str)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    engine = QueryEngine(DATA_ROOT, NET_PATH, device)

    h, w = args.height, args.width
    scale = config.img_size[0] / h

    data = get_data(args.query_json_dir, args.mixamo, scale, args.person, start=args.start, end=args.end)
    motion_input = data['input']

    since = time.time()
    result = engine.query(motion_input, require_view=args.require_view, top_n=args.top_n)
    print('query time:{}'.format(time.time() - since))
    print('top {} results:'.format(args.top_n))
    for item in result:
        print(item)

    if args.write_json is not None:
        ensure_dir(args.write_json)
        path = os.path.join(args.write_json, 'result.json')
        with open(path, 'w') as f:
            json.dump(result, f)
    '''
    if args.vis_dir is not None:
        out_dir = args.vis_dir
        ensure_dir(out_dir)

        path = os.path.join(out_dir, 'result.json')
        with open(path, 'w') as f:
            json.dump(result, f)

        out_path = os.path.join(out_dir, 'query.mp4')
        #motion2video(data['raw'], out_path, h, w)

        for i, item in enumerate(result):
            src_path = item['info']['video_path']
            out_path = os.path.join(out_dir, 'res-top{}.mp4'.format(i))
            clip_and_save_video(src_path, out_path, item['interval'][0], item['interval'][1])
    '''