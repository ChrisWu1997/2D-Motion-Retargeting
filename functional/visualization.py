import numpy as np
import os
import cv2
import math
import imageio
from tqdm import tqdm


def interpolate_color(color1, color2, alpha):
    color_i = alpha * np.array(color1) + (1 - alpha) * np.array(color2)
    return color_i.tolist()

def two_pts_to_rectangle(point1, point2):
    X = [point1[1], point2[1]]
    Y = [point1[0], point2[0]]
    length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
    length = 5
    alpha = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
    beta = alpha - 90
    if beta <= -180:
        beta += 360
    p1 = (   int(point1[0] - length*math.cos(math.radians(beta)))    ,   int(point1[1] - length*math.sin(math.radians(beta)))   )
    p2 = (   int(point1[0] + length*math.cos(math.radians(beta)))    ,   int(point1[1] + length*math.sin(math.radians(beta)))   )
    p3 = (   int(point2[0] + length*math.cos(math.radians(beta)))    ,   int(point2[1] + length*math.sin(math.radians(beta)))   )
    p4 = (   int(point2[0] - length*math.cos(math.radians(beta)))    ,   int(point2[1] - length*math.sin(math.radians(beta)))   )
    return [p1,p2,p3,p4]


def rgb2rgba(color):
    return (color[0], color[1], color[2], 255)


def hex2rgb(hex, number_of_colors=3):
    h = hex
    rgb = []
    for i in range(number_of_colors):
        h = h.lstrip('#')
        hex_color = h[0:6]
        rgb_color = [int(hex_color[i:i+2], 16) for i in (0, 2 ,4)]
        rgb.append(rgb_color)
        h = h[6:]

    return rgb


def joints2image(joints_position, colors, transparency=False, H=512, W=512, imtype=np.uint8):

    limbSeq = [[0, 1], [1, 2], [1, 5], [1, 8], [2, 3], [3, 4], [5, 6], [6, 7], \
               [8, 9], [8, 13], [9, 10], [10, 11], [11, 12], [13, 14], [14, 15], [15, 16],
               [0, 17], [0, 18]]

    L = rgb2rgba(colors[0]) if transparency else colors[0]
    M = rgb2rgba(colors[1]) if transparency else colors[1]
    R = rgb2rgba(colors[2]) if transparency else colors[2]

    colors_joints = [M, M, L, L, L, R, R, \
              R, M, L, L, L, L, R, R, R,
              R, R, L] + [L] * 15 + [R] * 15

    colors_limbs = [M, L, R, M, L, L, R, \
              R, L, R, L, L, L, R, R, R, \
              R, R]

    if transparency:
        canvas = np.zeros(shape=(H, W, 4))
    else:
        canvas = np.ones(shape=(H, W, 3)) * 255
    hips = joints_position[8]
    neck = joints_position[1]
    torso_length = ((hips[1] - neck[1]) ** 2 + (hips[0] - neck[0]) ** 2) ** 0.5

    head_radius = int(torso_length/4.5)
    end_effectors_radius = int(torso_length/15)
    end_effectors_radius = 7
    joints_radius = 7

    cv2.circle(canvas, (int(joints_position[0][0]),int(joints_position[0][1])), head_radius, colors_joints[0], thickness=-1)

    for i in range(1,49):
        if i in (17, 18):
            continue
        elif i > 18:
            radius = 2
        else:
            radius = joints_radius
        cv2.circle(canvas, (int(joints_position[i][0]),int(joints_position[i][1])), radius, colors_joints[i], thickness=-1)

    stickwidth = 2

    for i in range(16):
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
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        alpha = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))

        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(alpha), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, colors_limbs[i])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        bb = bounding_box(canvas)
        canvas_cropped = canvas[:,bb[2]:bb[3], :]

    return [canvas.astype(imtype), canvas_cropped.astype(imtype)]


def bounding_box(img):
    a = np.where(img != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox


def pose2im_all(all_peaks, H=512, W=512):
    limbSeq = [[1, 2], [2, 3], [3, 4],                       # right arm
               [1, 5], [5, 6], [6, 7],                       # left arm
               [8, 9], [9, 10], [10, 11],                    # right leg
               [8, 12], [12, 13], [13, 14],                  # left leg
               [1, 0],                                       # head/neck
               [1, 8],                                       # body,
               [0, 15], #[15, 17],                            # head-eye
               [0, 16], #[16, 18],                            # eye-ear
               ]

    limb_colors = [[0, 60, 255], [0, 120, 255], [0, 180, 255],
                    [180, 255, 0], [120, 255, 0], [60, 255, 0],
                    [170, 255, 0], [85, 255, 0], [0, 255, 0],
                    [255, 170, 0], [255, 85, 0], [255, 0, 0],
                    [0, 85, 255],
                    [0, 0, 255],
                    [240, 32, 160], #[139, 26, 85],
                    [0, 127, 255], #[0, 102, 205],
                   ]

    joint_colors = [[85, 0, 255], [0, 0, 255], [0, 60, 255], [0, 120, 255], [0, 180, 255],
                    [180, 255, 0], [120, 255, 0], [60, 255, 0], [0, 0, 255],
                    [170, 255, 0], [85, 255, 0], [0, 255, 0],
                    [255, 170, 0], [255, 85, 0], [255, 0, 0],
                    [211, 0, 148], [0, 165, 255],
                    #[226, 43, 138], [0, 133, 205],
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
