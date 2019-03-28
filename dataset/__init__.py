from torch.utils.data import DataLoader
from dataset.datasets import MixamoDatasetForSkeleton, MixamoDatasetForView, MixamoDatasetForThree
import numpy as np


def get_dataloader(name, phase, config, batch_size=64, num_workers=4):
    if name == 'skeleton':
        dataset = MixamoDatasetForSkeleton(phase, config.data_dir)
    elif name == 'view':
        dataset = MixamoDatasetForView(phase, config.data_dir)
    else:
        dataset = MixamoDatasetForThree(phase, config.data_dir)

    if phase == 'Train':
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, worker_init_fn=lambda _: np.random.seed())
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    return dataloader