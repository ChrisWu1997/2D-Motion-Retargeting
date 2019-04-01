import os
import numpy as np


def main():
    data_root = './mixamo_data'
    for phase in ['train', 'test']:
        phase_dir = os.path.join(data_root, phase)
        character_names = os.listdir(phase_dir)

        len_frames = 64 if phase == 'train' else 120

        for char in character_names:
            char_dir = os.path.join(phase_dir, char)
            animation_names = os.listdir(char_dir)

            for anim in animation_names:
                anim_path = os.path.join(char_dir, anim, anim + '.npy')
                animation = np.load(anim_path)

                mot_dir = os.path.join(char_dir, anim, 'motions')
                if not os.path.exists(mot_dir):
                    os.makedirs(mot_dir)

                total_length = animation.shape[-1]
                nr_motions = total_length // (len_frames // 2) - 1
                for i in range(nr_motions):
                    save_path = os.path.join(mot_dir, '{}.npy'.format(i + 1))
                    window = animation[:, :, i * (len_frames // 2): i * (len_frames // 2) + len_frames]
                    np.save(save_path, window)
                    print(save_path, window.shape)


if __name__ == '__main__':
    main()
