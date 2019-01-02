import numpy as np
from common import config
import os
import json
from scipy.ndimage import gaussian_filter1d


def trans_motion(motion3d, is_target=False):
    # global orthonormal projection
    motion_proj = motion3d[:, [0, 2], :]  # (15, 2, 64)
    motion_proj[:, 1, :] = - motion_proj[:, 1, :]

    motion_proj = motion_proj * config.unit

    # neck and mid-hip
    motion_proj[1, :, :] = (motion_proj[2, :, :] + motion_proj[5, :, :]) / 2
    motion_proj[8, :, :] = (motion_proj[9, :, :] + motion_proj[12, :, :]) / 2

    # subtract centers to local coordinates
    centers = motion_proj[8, :, :]
    motion_proj = motion_proj - centers

    if is_target:
        centers_reset = centers - centers[:, 0].reshape(2, 1) \
                        + np.array([[config.img_size[0] // 2], [config.img_size[1] // 2]])
        motion_proj = motion_proj + centers_reset
        return motion_proj

    # adding velocity and remove root
    velocity = np.c_[np.zeros((2, 1)), centers[:, 1:] - centers[:, :-1]].reshape(1, 2, -1)
    motion_proj = np.r_[motion_proj[:8], motion_proj[9:], velocity]

    return motion_proj


def trans_motion2d(motion2d):
    centers = motion2d[8, :, :]
    motion_proj = motion2d - centers

    # adding velocity
    velocity = np.c_[np.zeros((2, 1)), centers[:, 1:] - centers[:, :-1]].reshape(1, 2, -1)
    motion_proj = np.r_[motion_proj[:8], motion_proj[9:], velocity]

    return motion_proj


def trans_motion_inv(motion, sx=256, sy=256, velocity=None):
    if velocity is None:
        velocity = motion[-1].copy()
    motion_inv = np.r_[motion[:8], np.zeros((1, 2, motion.shape[-1])), motion[8:-1]]

    # restore centre position
    sum = 0
    for i in range(velocity.shape[-1]):
        sum += velocity[:, i]
        velocity[:, i] = sum
    velocity += np.array([[sx], [sy]])

    return motion_inv + velocity.reshape(1, 2, -1)


def normalize_motion(motion, mean_pose, std_pose):
    """
    :param motion: (J, 2, T)
    :param mean_pose: (J, 2)
    :param std_pose: (J, 2)
    :return:
    """
    return (motion - mean_pose[:, :, np.newaxis]) / std_pose[:, :, np.newaxis]


def normalize_motion_inv(motion, mean_pose, std_pose):
    if len(motion.shape) == 2:
        motion = motion.reshape(-1, 2, motion.shape[-1])
    return motion * std_pose[:, :, np.newaxis] + mean_pose[:, :, np.newaxis]


def json2motion(json_dir, scale=1.0, is_mixamo=False, smooth=True):
    json_files = sorted(os.listdir(json_dir))
    json_files = [os.path.join(json_dir, x) for x in json_files]

    motion = []
    for path in json_files:
        with open(path) as f:
            jointDict = json.load(f)
            if is_mixamo:
                pose_joints = np.array(jointDict['pose_keypoints_3d']).reshape((-1, 3))[:15]
                face_joints = np.array(jointDict['face_keypoints_3d']).reshape((-1, 3))
                joint = np.r_[pose_joints, face_joints]
            else:
                joint = np.array(jointDict['people'][0]['pose_keypoints_2d']).reshape((-1, 3))[:17, :2]
                if len(motion) > 0:
                    joint[np.where(joint == 0)] = motion[-1][np.where(joint == 0)]
            motion.append(joint)

    motion = np.stack(motion, axis=2)
    if smooth:
        motion = gaussian_filter1d(motion, sigma=2, axis=-1)
    motion = motion * scale
    return motion