from dataset.datasets import MixamoDatasetForSkeleton, MixamoDatasetForView, MixamoDatasetForFull
from torch.utils.data import DataLoader
from dataset.base_dataset import get_meanpose
import numpy as np


def get_dataloader(phase, config, batch_size=64, num_workers=4):
    assert config.name is not None
    if config.name == 'skeleton':
        dataset = MixamoDatasetForSkeleton(phase, config)
    elif config.name == 'view':
        dataset = MixamoDatasetForView(phase, config)
    else:
        dataset = MixamoDatasetForFull(phase, config)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, worker_init_fn=lambda _: np.random.seed())
    # if phase == 'Train':
    #     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
    #                             num_workers=num_workers, worker_init_fn=lambda _: np.random.seed())
    # else:
    #     dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    return dataloader
