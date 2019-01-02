import numpy as np
from common import config
import os
import json
from scipy.ndimage import gaussian_filter1d
CAM_HEIGHT = config.unit * 1
CAM_DIST = config.unit * 6
CANVAS_DIST = config.unit * 0.5


def rotate_coordinates(local3d, view_angle):
    cx, cy, cz = np.cos(view_angle)
    sx, sy, sz = np.sin(view_angle)

    x = local3d[0]
    x_cpm = np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ], dtype='float')
    x = x.reshape(-1, 1)
    mat33_x = cx * np.eye(3) + sx * x_cpm + (1.0 - cx) * np.matmul(x, x.T)

    mat33_z = np.array([
        [cz, sz, 0],
        [-sz, cz, 0],
        [0, 0, 1]
    ], dtype='float')

    local3d = local3d @ mat33_x.T @ mat33_z
    return local3d


def rotation_matrix_along_axis(x, angle):
    cx = np.cos(angle)
    sx = np.sin(angle)
    x_cpm = np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ], dtype='float')
    x = x.reshape(-1, 1)
    mat33_x = cx * np.eye(3) + sx * x_cpm + (1.0 - cx) * np.matmul(x, x.T)
    return mat33_x


def trans_motion(motion3d, local3d, scale=1.0, h_ratio=1.0, d_ratio=1.0, c_ratio=1.0):
    # local perspective projection
    #motion_proj = local3d[[0, 2], :] @ motion3d  # (15, 2, 64)
    motion3d = motion3d * config.unit

    # neck and mid-hip
    motion3d[1, :, :] = (motion3d[2, :, :] + motion3d[5, :, :]) / 2
    motion3d[8, :, :] = (motion3d[9, :, :] + motion3d[13, :, :]) / 2

    char_center = motion3d[8, :, 0].copy()
    char_center[2] = 0
    normal = -local3d[1, :]
    cam_pos = char_center + normal * CAM_DIST * d_ratio
    cam_pos[2] = CAM_HEIGHT * h_ratio
    canvas_center = cam_pos - normal * CANVAS_DIST * c_ratio

    ray = motion3d - cam_pos.reshape(1, -1, 1)
    t = np.dot(canvas_center - cam_pos, normal) / np.sum(ray * normal.reshape(1, -1, 1), axis=1, keepdims=True)
    motion_proj = cam_pos.reshape(1, -1, 1) + t * ray
    motion_proj = motion_proj - canvas_center.reshape(1, -1, 1)
    motion_proj = local3d[[0, 2], :] @ motion_proj

    motion_proj[:, 1, :] = - motion_proj[:, 1, :]
    motion_proj = motion_proj * 8 * scale

    # subtract centers to local coordinates
    motion_proj = trans_motion2d(motion_proj)
    #motion_proj = motion_proj * 8

    #motion_inv = trans_motion_inv(motion_proj)
    #if (np.sum(motion_inv > 512) > 0 or np.sum(motion_inv < 0) > 0):
        #print(np.max(motion_proj), name)
    #    return None

    return motion_proj


def trans_motion2d(motion2d):
    centers = motion2d[8, :, :]
    motion_proj = motion2d - centers

    centers = centers - centers[:, 0].reshape(2, 1) \
            + np.array([[config.img_size[0] // 2], [config.img_size[1] // 2]])

    # adding velocity
    velocity = np.c_[np.zeros((2, 1)), centers[:, 1:] - centers[:, :-1]].reshape(1, 2, -1)
    motion_proj = np.r_[motion_proj[:8], motion_proj[9:], velocity]
    return motion_proj


def trans_motion_inv(motion, sx=256, sy=256, velocity=None):
    if velocity is None:
        velocity = motion[-1].copy()
    motion_inv = np.r_[motion[:8], np.zeros((1, 2, motion.shape[-1])), motion[8:-1]]

    # restore centre position
    centers = np.zeros_like(velocity)
    sum = 0
    for i in range(motion.shape[-1]):
        sum += velocity[:, i]
        centers[:, i] = sum
    centers += np.array([[sx], [sy]])

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


def json2motion(json_dir, scale=1.0, is_mixamo=False, is_our_output=False, smooth=True):
    json_files = sorted(os.listdir(json_dir))
    json_files = [os.path.join(json_dir, x) for x in json_files]

    motion = []
    for path in json_files:
        with open(path) as f:
            jointDict = json.load(f)
            if is_mixamo:
                pose_joints = np.array(jointDict['pose_keypoints_3d']).reshape((-1, 3))[:17, :]
                face_joints = np.array(jointDict['face_keypoints_3d']).reshape((-1, 3))
                hand_joints = np.array(jointDict['hand_keypoints_3d']).reshape((-1, 3))
                joint = np.r_[pose_joints, face_joints, hand_joints]
            elif is_our_output:
                all_joints = np.array(jointDict['pose_keypoints']).reshape((-1, 2))
                pose_joints = all_joints[0:17, :]
                rhand_joints = all_joints[17:32, :]
                lhand_joints = all_joints[32:49, :]
                joint = np.r_[pose_joints, rhand_joints, lhand_joints]
                if len(motion) > 0:
                    joint[np.where(joint == 0)] = motion[-1][np.where(joint == 0)]
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

    motion = np.stack(motion, axis=2)
    if smooth:
        motion = gaussian_filter1d(motion, sigma=2, axis=-1)
    motion = motion * scale
    return motion


if __name__ == '__main__':
    local3d = np.load('/data1/wurundi/mixamo/mixamo-3d-data/Jasper-full/Back-Squat/locals/1.npy')
    x = local3d[0]
    print(np.sum(x ** 2))

    print(local3d)
    angle = (np.pi / 9 , 0, 0)
    local3d = rotate_coordinates(local3d, angle)
    print(local3d)
