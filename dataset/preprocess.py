import os
import numpy as np
import json
import shutil

if __name__ == '__main__':
    data_root = '/data1/wurundi/mixamo/mixamo-3d-data'
    len_frames = 64

    character_names = ['Jasper-full']#['Andromeda', 'Pumpkinhulk', 'SportyGranny', 'Whiteclown']
    #['Aj', 'BigVegas', 'Claire', 'Jasper', 'Lola', 'Malcolm', 'Pearl', 'Warrok']
    #['BigVegas', 'Warrok'] #['Andromeda', 'SportyGranny', 'Pumpkinhulk', 'Whiteclown'] #['Aj', 'Claire', 'Jasper', 'Kaya', 'Lola', 'Malcolm', 'Pearl']
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

            nr_motions = nr_files // (len_frames // 2) - 1
            total_num += nr_motions
            for i in range(nr_motions):
                save_dir = os.path.join(mot_dir, '{}.npy'.format(i + 1))
                window = motion[:, :, i * (len_frames // 2): i * (len_frames // 2) + len_frames]
                np.save(save_dir, window)
                print(save_dir, window.shape)

