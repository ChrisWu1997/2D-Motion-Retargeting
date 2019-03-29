from model.networks import AutoEncoder2x, AutoEncoder3x
import torch.nn as nn
import torch.nn.functional as F


def get_autoencoder(config):
    assert config.name is not None
    if config.name == 'skeleton':
        return AutoEncoder2x(config.mot_en_channels, config.body_en_channels, config.de_channels,
                             global_pool=F.max_pool1d, convpool=nn.MaxPool1d, compress=False)
    elif config.name == 'view':
        return AutoEncoder2x(config.mot_en_channels, config.view_en_channels, config.de_channels,
                             global_pool=F.max_pool1d, convpool=nn.MaxPool1d, compress=False)   #FIXME: max/avg
    else:
        return AutoEncoder3x(config.mot_en_channels, config.body_en_channels,
                             config.view_en_channels, config.de_channels)
