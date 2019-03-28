import os
import json
import numpy as np


if __name__ == '__main__':
    # data_root = '/data1/wurundi/mixamo/mixamo-3d-data'
    data_root = '/data1/wurundi/mixamo/validation'
    # out_root = '/data1/wurundi/mixamo_release' + '/train'
    out_root = '/data1/wurundi/mixamo_release' + '/test'

    if not os.path.exists(out_root):
        os.makedirs(out_root)

    # character_names = ['Aj', 'BigVegas', 'Claire', 'Jasper', 'Lola', 'Malcolm',
    #                    'Pearl', 'Warrok', 'Globin', 'Kaya', 'PeanutMan']
    character_names = ['Andromeda', 'Pumpkinhulk', 'SportyGranny', 'Ty']
    character_names = [x for x in character_names if os.path.isdir(os.path.join(data_root, x))]

    print(character_names)

    total_num = 0
    for char in character_names:
        char_dir = os.path.join(data_root, char)
        animation_names = os.listdir(char_dir)

        for anim in animation_names:
            joint_dir = os.path.join(char_dir, anim, 'jointsDict')
            mot_dir = os.path.join(char_dir, anim, 'motions')

            if not os.path.exists(mot_dir):
                os.makedirs(mot_dir)

            nr_files = len(os.listdir(joint_dir))

            motion = []
            for i in range(0, nr_files):
                with open(os.path.join(joint_dir, '%04d_keypoints.json' % i)) as f:
                    joint = json.load(f)
                    joint = np.array(joint['people'][0]['pose_keypoints_2d']).reshape((-1, 3))
                motion.append(joint[:15, :])

            motion = np.stack(motion, axis=2)

            out_dir = os.path.join(out_root, char, anim)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            path = os.path.join(out_dir, anim + '.npy')
            np.save(path, motion)

        print("{} finished".format(char))
