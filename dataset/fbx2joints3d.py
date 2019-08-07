# This script is used for automatically converting fbx files to 3d joint positions and
# save it as .json files in OpenPose style.
# Although this script is orignially designed to deal with the bunch of downloaded mixamo data,
# but you can easily modify it for your own purpose.
#
# IMPORTANT!!! Please install blender before running this script. https://www.blender.org
#
# INSTRUCTIONS:
# 1. Specify SRC_DATA_DIR, the directory for mixamo fbx files, which should follow the below structure:
#    SRC_DATA_DIR ---- | character1_dir
#                      | character2_dir ---- | animation1.fbx
#                      | ...                 | animation2.fbx
#                                            | animation3.fbx
#                                            | ...
# 2. Specify OUT_DATA_DIR, the directory for output .json files, which would be the below structure when finishing:
#    OUT_DATA_DIR ---- | character1_dir
#                      | character2_dir ---- | animation1
#                      | ...                 | animation2 ---- | jointsDict ---- | 0000_keypoints.json
#                                            | ...                               | 0001_keypoints.json
#                                                                                | ...
# 3. Specify MIN_NR_FRAMES. Fbx file that contains less than MIN_NR_FRAMES frames will be discarded. Default 64.
# 4. Specify BASE_JOINT_NAMES, which joints you wish to use. Default 15 joints listed below.
# 5. Run as blender python script. Command: 'blender --background -P fbx2joints3d.py'.

import bpy
import os
import time
import sys
import json
from mathutils import Vector

HOME_FILE_PATH = os.path.abspath('./homefile.blend')
SRC_DATA_DIR = "/data1/wurundi/mixamo/mixamo"
OUT_DATA_DIR = "/data1/wurundi/mixamo/mixamo-3d-data"
MIN_NR_FRAMES = 64
RESOLUTION = (512, 512)

BASE_JOINT_NAMES = ['Head', 'Neck',
                    'RightArm', 'RightForeArm', 'RightHand', 'LeftArm', 'LeftForeArm', 'LeftHand',
                    'Hips', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot',
                    ]


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_homefile(filepath):
    cube = bpy.data.objects['Cube']
    bpy.data.objects.remove(cube)

    bpy.data.objects['Lamp'].data.energy = 2
    bpy.data.objects['Lamp'].data.type = 'HEMI'

    bpy.data.scenes['Scene'].render.resolution_x = RESOLUTION[0]
    bpy.data.scenes['Scene'].render.resolution_y = RESOLUTION[1]
    bpy.data.scenes['Scene'].render.resolution_percentage = 100

    bpy.data.worlds['World'].use_sky_blend = True
    bpy.data.worlds['World'].horizon_color = (1, 1, 1)
    bpy.data.worlds['World'].zenith_color = (1, 1, 1)

    bpy.ops.wm.save_as_mainfile(filepath=filepath)


def clear_scene_and_import_fbx(filepath):
    """
    Clear the whole scene and import fbx file into the empty scene.

    :param filepath: filepath for fbx file
    """
    # redirect blender output info
    logfile = 'blender_render.log'
    open(logfile, 'w').close()
    old = os.dup(1)
    sys.stdout.flush()
    os.close(1)
    os.open(logfile, os.O_WRONLY)

    bpy.ops.wm.read_homefile(filepath=HOME_FILE_PATH)
    bpy.ops.import_scene.fbx(filepath=filepath)

    os.close(1)
    os.dup(old)
    os.close(old)


def get_joint3d_positions(joint_names, frame_idx):
    """
    Get joint3d positions for current armature and given frame index.

    :param frame_idx: frame index
    :param joint_names: list of joint names
    :return: dict, {'pose_keypoints_3d': [x1, y1, z1, x2, y2, z2, ...]}
    """
    bpy.context.scene.frame_set(frame_idx)

    posebones = bpy.data.objects['Armature'].pose.bones
    armature = bpy.data.objects['Armature']

    out_dict = {'pose_keypoints_3d': []}
    for name in joint_names:
        global_location = armature.matrix_world * posebones[name].matrix * Vector((0, 0, 0))
        l = [global_location[0], global_location[1], global_location[2]]
        out_dict['pose_keypoints_3d'].extend(l)
    return out_dict


def main():
    set_homefile(HOME_FILE_PATH)

    ensure_dir(OUT_DATA_DIR)

    character_names = os.listdir(SRC_DATA_DIR) # ['Jasper']

    for char in character_names:
        fbx_dir = os.path.join(SRC_DATA_DIR, char)
        files = os.listdir(fbx_dir)

        if char == 'Ty':
            joint_names = ['Boy:' + x for x in BASE_JOINT_NAMES]
        elif char == 'Swat':
            joint_names = ['swat:' + x for x in BASE_JOINT_NAMES]
        elif char == 'BigVegas':
            joint_names = ['newVegas:' + x for x in BASE_JOINT_NAMES]
        elif char in ['Andromeda', 'Douglas', 'Jasper', 'Liam', 'Malcolm', 'Pearl', 'Remy', 'Stefani']:
            joint_names = ['mixamorig:' + x for x in BASE_JOINT_NAMES]
        else:
            joint_names = BASE_JOINT_NAMES

        for j, name in enumerate(files):
            path = os.path.join(fbx_dir, name)
            since = time.time()

            animation_name = name.split('.')[0]

            clear_scene_and_import_fbx(path)

            frame_end = bpy.data.actions[0].frame_range[1]
            if frame_end < MIN_NR_FRAMES - 1:
                continue

            save_dir = os.path.join(OUT_DATA_DIR, char, animation_name, 'jointsDict')
            ensure_dir(save_dir)

            # iteratively get joint3d position for each frame
            for i in range(int(frame_end) + 1):
                out_dict = get_joint3d_positions(joint_names, i)

                save_path = os.path.join(save_dir, '%04d_keypoints.json' % i)
                with open(save_path, 'w') as f:
                    json.dump(out_dict, f)

            process_time = time.time() - since

            print('[{}/{}]({:.2f}) {}'.format(j, len(files), process_time, path))


def json2npy():
    data_root = OUT_DATA_DIR

    character_names = os.listdir(data_root)
    print(character_names)

    total_num = 0
    for char in character_names:
        char_dir = os.path.join(data_root, char)
        animation_names = os.listdir(char_dir)

        for anim in animation_names:
            joint_dir = os.path.join(char_dir, anim, 'jointsDict')

            nr_files = len(os.listdir(joint_dir))
            motion = []
            for i in range(0, nr_files):
                with open(os.path.join(joint_dir, '%04d_keypoints.json' % i)) as f:
                    info = json.load(f)
                    joint = np.array(info['pose_keypoints_3d']).reshape((-1, 3))
                motion.append(joint[:15, :])

            motion = np.stack(motion, axis=2)
            save_path = os.path.join(char_dir, anim, '{}.npy'.format(anim))
            print(save_path)
            np.save(save_path, motion)


if __name__ == '__main__':
    main()
    # further convert json to npy
    json2npy()
