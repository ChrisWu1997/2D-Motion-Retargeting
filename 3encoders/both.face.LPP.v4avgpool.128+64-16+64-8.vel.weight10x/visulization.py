import numpy as np
import os
import cv2
import math
from cluster import cluster_motion, cluster_view, cluster_body
import torch
import torch.nn as nn
from motion import trans_motion_inv, normalize_motion_inv, trans_motion
import imageio
from tqdm import tqdm


def visulize_motion_in_training(motion, mean_pose, std_pose, nr_visual=8, H=512, W=512):
    inds = np.linspace(0, motion.shape[1] - 1, nr_visual, dtype=int)
    motion = motion[:, inds]
    motion = motion.reshape(-1, 2, motion.shape[-1])
    motion = normalize_motion_inv(motion, mean_pose, std_pose)
    peaks = trans_motion_inv(motion)

    heatmaps = []
    for i in range(peaks.shape[2]):
        skeleton = pose2im_all(peaks[:, :, i], H, W)
        heatmaps.append(skeleton)
    heatmaps = np.stack(heatmaps).transpose((0, 3, 1, 2)) / 255.0
    return heatmaps


def pose2im_all(all_peaks, H=512, W=512):
    limbSeq = [[1, 2], [2, 3], [3, 4],                       # right arm
               [1, 5], [5, 6], [6, 7],                       # left arm
               [8, 9], [9, 10], [10, 11],                    # right leg
               [8, 12], [12, 13], [13, 14],                  # left leg
               [1, 0],                                       # head/neck
               [1, 8],                                       # body,
               ]

    limb_colors = [[0, 60, 255], [0, 120, 255], [0, 180, 255],
              [180, 255, 0], [120, 255, 0], [60, 255, 0],
              [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [255, 170, 0], [255, 85, 0], [255, 0, 0],
              [0, 85, 255],
              [0, 0, 255],
                   ]

    joint_colors = [[85, 0, 255], [0, 0, 255], [0, 60, 255], [0, 120, 255], [0, 180, 255],
                    [180, 255, 0], [120, 255, 0], [60, 255, 0], [0, 0, 255],
                    [170, 255, 0], [85, 255, 0], [0, 255, 0],
                    [255, 170, 0], [255, 85, 0], [255, 0, 0],
                    ]

    image = pose2im(all_peaks, limbSeq, limb_colors, joint_colors, H, W)
    return image


def pose2im(all_peaks, limbSeq, limb_colors, joint_colors, H, W, _circle=True, _limb=True, imtype=np.uint8):
    canvas = np.zeros(shape=(H, W, 3))
    canvas.fill(255)

    if _circle:
        for i in range(len(joint_colors)):
            cv2.circle(canvas, (int(all_peaks[i][0]), int(all_peaks[i][1])), 2, joint_colors[i], thickness=2)

    if _limb:
        stickwidth = 2

        for i in range(len(limbSeq)):
            limb = limbSeq[i]
            cur_canvas = canvas.copy()
            point1_index = limb[0]
            point2_index = limb[1]

            if len(all_peaks[point1_index]) > 0 and len(all_peaks[point2_index]) > 0:
                point1 = all_peaks[point1_index][0:2]
                point2 = all_peaks[point2_index][0:2]
                X = [point1[1], point2[1]]
                Y = [point1[0], point2[0]]
                mX = np.mean(X)
                mY = np.mean(Y)
                # cv2.line()
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(cur_canvas, polygon, limb_colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas.astype(imtype)


def joints2image(joints_position, H=512, W=512, imtype=np.uint8):

    limbSeq = [[0, 1], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7], \
               [8, 9], [8, 12], [9, 10], [10, 11], [12, 13], [13, 14], [0, 15], \
               [0, 16]]

    M = [222,119,138]
    L = [212,66,128]
    R = [102,31,70]
    BG = [255,255,255]

    colors_joints = [M, M, L, L, L, R, R, \
              R, M, L, L, L, R, R, \
              R, R]

    colors_limbs = [M, L, R, M, L, L, R, \
              R, L, R, L, L, R, R, \
              R, R]

    canvas = np.zeros(shape=(H, W, 3))
    canvas[:,:,0] = BG[0]
    canvas[:,:,1] = BG[1]
    canvas[:,:,2] = BG[2]
    hips = joints_position[8]
    neck = joints_position[1]
    torso_length = ((hips[1] - neck[1]) ** 2 + (hips[0] - neck[0]) ** 2) ** 0.5

    head_radius = int(torso_length/4.5)
    end_effectors_radius = int(torso_length/15)
    end_effectors_radius = 7
    joints_radius = 7

    cv2.circle(canvas, (int(joints_position[0][0]),int(joints_position[0][1])), head_radius, colors_joints[0], thickness=-1)

    for i in range(1,15):
        if i in [4,7,11,14]:
            radius = end_effectors_radius
        else:
            radius = joints_radius
        cv2.circle(canvas, (int(joints_position[i][0]),int(joints_position[i][1])), radius, colors_joints[i], thickness=-1)

    stickwidth = 2

    for i in range(14):
        limb = limbSeq[i]
        cur_canvas = canvas.copy()
        point1_index = limb[0]
        point2_index = limb[1]

        #if len(all_peaks[point1_index]) > 0 and len(all_peaks[point2_index]) > 0:
        point1 = joints_position[point1_index]
        point2 = joints_position[point2_index]
        X = [point1[1], point2[1]]
        Y = [point1[0], point2[0]]
        mX = np.mean(X)
        mY = np.mean(Y)
        # cv2.line()
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        alpha = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        # beta = alpha - 90
        # if beta <= -180:
        #     beta += 360
        # print("limb: ", i)
        # print("alpha: ", alpha)
        # p = two_pts_to_rectangle(point1, point2)
        # for j in range(4):
        #     cv2.line(canvas, p[j], p[(j+1)%4], colors_limbs[i] )

        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(alpha), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, colors_limbs[i])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas.astype(imtype)


def cluster_in_training(net, cluster_data, mode, device, cluster_dir, epoch):
    is_draw = epoch % 50 == 0
    out = {}

    path = os.path.join(cluster_dir, '{}.body/{}.png'.format(mode, epoch))
    sil_score, img = cluster_body(net, cluster_data, device, path, is_draw=is_draw)

    out['cluster_body'] = img
    out['silhouette_score_body'] = sil_score

    path = os.path.join(cluster_dir, '{}.view/{}.png'.format(mode, epoch))
    sil_score, img = cluster_view(net, cluster_data, device, path, is_draw=is_draw)

    out['cluster_view'] = img
    out['silhouette_score_view'] = sil_score

    path = os.path.join(cluster_dir, '{}.motion/{}_both.png'.format(mode, epoch))
    sil_score, img = cluster_motion(net, cluster_data, device, path, is_draw=is_draw)
    out['cluster_motion_both'] = img
    out['silhouette_score_motion_both'] = sil_score

    path = os.path.join(cluster_dir, '{}.motion/{}_body.png'.format(mode, epoch))
    sil_score, img = cluster_motion(net, cluster_data, device, path, is_draw=is_draw, mode='body')
    out['cluster_motion_body'] = img
    out['silhouette_score_motion_body'] = sil_score

    path = os.path.join(cluster_dir, '{}.motion/{}_view.png'.format(mode, epoch))
    sil_score, img = cluster_motion(net, cluster_data, device, path, is_draw=is_draw, mode='view')
    out['cluster_motion_view'] = img
    out['silhouette_score_motion_view'] = sil_score

    return out


def visualize_filters(net, save_dir, gamma=1.5):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # motion part
    cnt = 0
    for layer in net.mot_encoder.model:
        if isinstance(layer, nn.Conv1d):
            cnt += 1
            weight = layer.weight.detach().cpu().numpy()
            for i in range(weight.shape[0]):
                filter = weight[i]
                filter = np.abs(filter)
                filter_norm = (filter / (np.max(filter) - np.min(filter))) ** gamma * 255
                filter_norm = filter_norm.astype(np.uint8)
                img_color = cv2.applyColorMap(filter_norm, 2)
                save_path = os.path.join(save_dir, 'motion-conv{}-{}.png'.format(cnt, i))
                cv2.imwrite(save_path, img_color)

    cnt = 0
    for layer in net.body_encoder.model:
        if isinstance(layer, nn.Conv1d):
            cnt += 1
            weight = layer.weight.detach().cpu().numpy()
            for i in range(weight.shape[0]):
                filter = weight[i]
                filter = np.abs(filter)
                filter_norm = (filter / (np.max(filter) - np.min(filter))) ** gamma * 255
                filter_norm = filter_norm.astype(np.uint8)
                img_color = cv2.applyColorMap(filter_norm, 2)
                save_path = os.path.join(save_dir, 'other-conv{}-{}.png'.format(cnt, i))
                cv2.imwrite(save_path, img_color)


def motion2video(motion, h, w, save_path):
    videowriter = imageio.get_writer(save_path, fps=25)
    vlen = motion.shape[-1]
    for i in tqdm(range(vlen)):
        img = joints2image(motion[:, :, i], H=h, W=w)
        videowriter.append_data(img)
    videowriter.close()


if __name__ == '__main__':
    '''
    meanpose = np.load('/data1/wurundi/mixamo/mixamo-3d-data/meanpose_whole_7view-LOP-noroot-gt.npy')
    meanpose = trans_motion_inv(meanpose[:, :, np.newaxis])
    meanpose = meanpose[:, :, 0]
    img = pose2im_all(meanpose)
    cv2.imwrite('./test.png', img)
    '''
    TRAIN_CHARACTERS = ['Aj', 'BigVegas', 'Claire', 'Jasper', 'Lola', 'Malcolm', 'Pearl', 'Warrok', 'Globin', 'Kaya',
                        'PeanutMan']
    for char in TRAIN_CHARACTERS:
        motion3d = np.load('/data1/wurundi/mixamo/mixamo-3d-data/{}/Quarterback-Pass/motions/2.npy'.format(char))
        local3d = np.load('/data1/wurundi/mixamo/mixamo-3d-data/{}/Quarterback-Pass/locals/2.npy'.format(char))
        motion_proj = trans_motion_inv(trans_motion(motion3d, local3d, 1.0, 1.0, 1.0))

        videowriter = imageio.get_writer('./assets/{}.mp4'.format(char), fps=25)
        vlen = motion_proj.shape[-1]
        for i in tqdm(range(vlen)):
            img = pose2im_all(motion_proj[:, :, i])
            videowriter.append_data(img)
        videowriter.close()
