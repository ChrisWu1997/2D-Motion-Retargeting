from PIL import Image
import os
import json
import torch
import numpy as np
import numbers
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import cv2
import logging
import math
import shutil
import csv

class TrainClock(object):
    def __init__(self):
        self.epoch = 1
        self.minibatch = 0
        self.step = 0

    def tick(self):
        self.minibatch += 1
        self.step += 1

    def tock(self):
        self.epoch += 1
        self.minibatch = 0

    def make_checkpoint(self):
        return {
            'epoch': self.epoch,
            'minibatch': self.minibatch,
            'step': self.step
        }

    def restore_checkpoint(self, clock_dict):
        self.epoch = clock_dict['epoch']
        self.minibatch = clock_dict['minibatch']
        self.step = clock_dict['step']


class Table(object):
    def __init__(self, filename):
        '''
        create a table to record experiment results that can be opened by excel
        :param filename: using '.csv' as postfix
        '''
        assert '.csv' in filename
        self.filename = filename

    @staticmethod
    def merge_headers(header1, header2):
        #return list(set(header1 + header2))
        if len(header1) > len(header2):
            return header1
        else:
            return header2

    def write(self, ordered_dict):
        '''
        write an entry
        :param ordered_dict: something like {'name':'exp1', 'acc':90.5, 'epoch':50}
        :return:
        '''
        if os.path.exists(self.filename) == False:
            headers = list(ordered_dict.keys())
            prev_rec = None
        else:
            with open(self.filename) as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                prev_rec = [row for row in reader]
            headers = self.merge_headers(headers, list(ordered_dict.keys()))

        with open(self.filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, headers)
            writer.writeheader()
            if not prev_rec == None:
                writer.writerows(prev_rec)
            writer.writerow(ordered_dict)


class WorklogLogger:
    def __init__(self, log_file):
        logging.basicConfig(filename=log_file,
                            level=logging.DEBUG,
                            format='%(asctime)s - %(threadName)s -  %(levelname)s - %(message)s')

        self.logger = logging.getLogger()

    def put_line(self, line):
        self.logger.info(line)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_args(args, save_dir):
    param_path = os.path.join(save_dir, 'params.json')

    with open(param_path, 'w') as fp:
        json.dump(args.__dict__, fp, indent=4, sort_keys=True)


def ensure_dir(path):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def ensure_dirs(paths):
    """
    create paths by first checking their existence
    :param paths: list of path
    :return:
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            ensure_dir(path)
    else:
        ensure_dir(paths)


def remkdir(path):
    """
    if dir exists, remove it and create a new one
    :param path:
    :return:
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def motion2json(motion, h, w, save_dir):
    nr_frames = motion.shape[-1]
    for i in range(nr_frames):
        path = os.path.join(save_dir, "%04d_keypoints.json" % i)
        out_dict = {"resolution": (h, w), 'pose_keypoints': motion[:, :, i].reshape(-1).tolist()}
        with open(path, 'w') as f:
            json.dump(out_dict, f)

def motion2openpose(motion, h, w, save_dir, is_hand=True):
    """

    :param motion: (NR_JOINTS, 2, NR_FRAMES)
    :param h:
    :param w:
    :param save_dir:
    :return:
    """
    if is_hand:
        pose_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8,
                9, 10, 11, 22, 12, 13, 14, 19,
                15, 16]
        hand_idx = [1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20]
    else: # basic pose + two eyes
        pose_idx = np.arange(17)
    nr_frames = motion.shape[-1]
    for i in range(nr_frames):
        path = os.path.join(save_dir, "%04d_keypoints.json" % i)
        out_dict = {"resolution": (h, w),
                    "people":[{"pose_keypoints_2d":[0] * 66, "face_keypoints_2d":[],
                               "hand_left_keypoints_2d":[0] * 45, "hand_right_keypoints_2d":[0] * 45}]}
        pose_joints = np.zeros((25, 3))
        pose_joints[pose_idx, :2] = motion[:len(pose_idx), :, i]
        out_dict['people'][0]['pose_keypoints_2d'] = pose_joints.reshape(-1).tolist()
        if is_hand:
            rhand_joints = np.zeros((21, 3))
            rhand_joints[hand_idx, :2] = motion[19:34, :, i]
            lhand_joints = np.zeros((21, 3))
            lhand_joints[hand_idx, :2] = motion[34:49, :, i]
            out_dict['people'][0]['hand_right_keypoints_2d'] = rhand_joints.reshape(-1).tolist()
            out_dict['people'][0]['hand_left_keypoints_2d'] = lhand_joints.reshape(-1).tolist()

        with open(path, 'w') as f:
            json.dump(out_dict, f)


def test():
    pass


if __name__ == '__main__':
    test()
