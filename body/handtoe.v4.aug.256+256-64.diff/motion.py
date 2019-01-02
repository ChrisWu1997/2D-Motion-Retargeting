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
    motion_proj[8, :, :] = (motion_proj[9, :, :] + motion_proj[13, :, :]) / 2

    motion_proj = trans_motion2d(motion_proj)

    return motion_proj


def trans_motion2d(motion2d):
    centers = motion2d[8, :, :]
    motion_proj = motion2d - centers

    # adding difference
    difference = (centers - centers[:, 0].reshape(2, 1)).reshape(1, 2, -1)
    motion_proj = np.r_[motion_proj[:8], motion_proj[9:], difference]

    return motion_proj


def trans_motion_inv(motion, sx=256, sy=256, difference=None):
    if difference is None:
        difference = motion[-1].copy()
    motion_inv = np.r_[motion[:8], np.zeros((1, 2, motion.shape[-1])), motion[8:-1]]

    # restore centre position
    centers = difference + np.array([[sx], [sy]])

    return motion_inv + centers.reshape(1, 2, -1)


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


pose_idx = [0, 1,  2, 3, 4,  5, 6, 7,  8,
            9, 10, 11, 22,  12, 13, 14, 19,
            15, 16]
hand_idx = [1, 2, 4,  5, 6, 8,  9, 10, 12,  13, 14, 16,  17, 18, 20]


def json2motion(json_dir, scale=1.0, is_mixamo=False, smooth=True, length=float('inf')):
    json_files = sorted(os.listdir(json_dir))
    json_files = [os.path.join(json_dir, x) for x in json_files]

    motion = []
    for i, path in enumerate(json_files):
        with open(path) as f:
            jointDict = json.load(f)
            if is_mixamo:
                pose_joints = np.array(jointDict['pose_keypoints_3d']).reshape((-1, 3))[:17, :]
                face_joints = np.array(jointDict['face_keypoints_3d']).reshape((-1, 3))
                hand_joints = np.array(jointDict['hand_keypoints_3d']).reshape((-1, 3))
                joint = np.r_[pose_joints, face_joints, hand_joints]
            else:
                pose_joints = np.array(jointDict['people'][0]['pose_keypoints_2d']).reshape((-1, 3))
                rhand_joints = np.array(jointDict['people'][0]['hand_right_keypoints_2d']).reshape((-1, 3))
                lhand_joints = np.array(jointDict['people'][0]['hand_left_keypoints_2d']).reshape((-1, 3))
                pose_joints = pose_joints[pose_idx, :2]
                rhand_joints = rhand_joints[hand_idx, :2]
                lhand_joints = lhand_joints[hand_idx, :2]
                joint = np.r_[pose_joints, rhand_joints, lhand_joints]
                if len(motion) > 0:
                    joint[np.where(joint == 0)] = motion[-1][np.where(joint == 0)]
            motion.append(joint)
        if i == (length-1):
            break
    motion = np.stack(motion, axis=2)
    if smooth:
        motion = gaussian_filter1d(motion, sigma=2, axis=-1)
    motion = motion * scale
    return motion
