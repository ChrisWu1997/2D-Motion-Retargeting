import numpy as np
from common import config
import os
import json
from scipy.ndimage import gaussian_filter1d

def rotate_coordinates(local3d, view_angle):
    cx, cy, cz = np.cos(view_angle)
    sx, sy, sz = np.sin(view_angle)
    mat33_x = np.array([
        [1, 0, 0],
        [0, cx, sx],
        [0, -sx, cx]
    ], dtype='float')
    mat33_z = np.array([
        [cz, sz, 0],
        [-sz, cz, 0],
        [0, 0, 1]
    ], dtype='float')

    local3d = local3d @ mat33_x @ mat33_z
    return local3d


def trans_motion(motion3d, local3d, is_target=False):
    # global orthonormal projection
    motion3d[:, 2, :] = - motion3d[:, 2, :]

    motion3d = motion3d * config.unit
    motion_proj = local3d[[0, 2], :] @ motion3d  # (15, 2, 64)
    #motion_proj[:, 1, :] = - motion_proj[:, 1, :]

    #motion_proj = motion_proj * config.unit

    # subtract centers to local coordinates
    motion_proj, centers = trans_motion2d(motion_proj)

    if is_target:
        return motion_proj + centers

    return motion_proj, centers


def trans_motion2d(motion2d):
    centers = motion2d[8, :, :]
    motion_proj = motion2d - centers

    centers = centers - centers[:, 0].reshape(2, 1) \
            + np.array([[config.img_size[0] // 2], [config.img_size[1] // 2]])

    motion_proj = np.r_[motion_proj[:8], motion_proj[9:]]
    return motion_proj, centers


def trans_motion_inv(motion, centers):
    motion_inv = np.r_[motion[:8], np.zeros((1, 2, motion.shape[-1])), motion[8:]]
    return motion_inv + centers


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


def json2motion(json_list, is_mixamo=False, person=0):
    mot = []
    for filename in json_list:
        with open(filename) as f:
            joint = json.load(f)
            if is_mixamo:
                joint = np.array(joint['people'][person]['pose_keypoints_2d']).reshape((-1, 3))[:15, :]
            else:
                if len(mot) > 0:
                    mark_idx = -1
                    min_diff = 100000
                    nr_people = len(joint['people'])
                    for j in range(nr_people):
                        cur_joint = np.array(joint['people'][j]['pose_keypoints_2d']).reshape((-1, 3))[:15, :2]
                        cur_joint[np.where(cur_joint == 0)] = mot[-1][np.where(cur_joint == 0)]
                        cur_diff = np.mean(np.sqrt(np.sum((cur_joint - mot[-1]) ** 2, axis=1)))
                        if cur_diff < min_diff:
                            # print("not temporal coherent")
                            min_diff = cur_diff
                            mark_idx = j
                    joint = np.array(joint['people'][mark_idx]['pose_keypoints_2d']).reshape((-1, 3))[:15, :2]
                else:
                    joint = np.array(joint['people'][person]['pose_keypoints_2d']).reshape((-1, 3))[:15, :2]
            mot.append(joint)

    mot = np.stack(mot, axis=2)
    mot = gaussian_filter1d(mot, sigma=2, axis=-1)
    if is_mixamo:
        mot = mot[:, [0, 2], :]  # (15, 2, 64)
        mot[:, 1, :] = - mot[:, 1, :]

        mot = mot * config.unit

        centers = mot[8, :, :]
        mot = mot - centers
        centers_reset = centers - centers[:, 0].reshape(2, 1) \
                        + np.array([[config.img_size[0] // 2], [config.img_size[1] // 2]])
        mot = mot + centers_reset
    return mot
