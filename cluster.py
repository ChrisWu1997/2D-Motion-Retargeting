import sys
sys.path.append("..")
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.switch_backend('agg')
import torch
from dataset import MixamoDatasetForFull
from common import config
from model import get_autoencoder
import cv2
import time
import argparse
import os


def tsne_on_pca(arr, is_PCA=True):
    """
    visualize through t-sne on pca reduced data
    :param arr: (nr_examples, nr_features)
    :return:
    """
    if is_PCA:
        pca_50 = PCA(n_components=50)
        arr = pca_50.fit_transform(arr)
    tsne_2 = TSNE(n_components=2)
    res = tsne_2.fit_transform(arr)
    return res


def cluster_body(net, cluster_data, device, save_path):
    data, characters = cluster_data[0], cluster_data[2]
    data = data[:, :, 0, :, :]
    # data = data.reshape(-1, data.shape[2], data.shape[3], data.shape[4])

    nr_mv, nr_char = data.shape[0], data.shape[1]
    labels = np.arange(0, nr_char).reshape(1, -1)
    labels = np.tile(labels, (nr_mv, 1)).reshape(-1)
    
    if hasattr(net, 'static_encoder'):
        features = net.static_encoder(data.contiguous().view(-1, data.shape[2], data.shape[3])[:, :-2, :].to(device))
    else:
        features = net.body_encoder(data.contiguous().view(-1, data.shape[2], data.shape[3])[:, :-2, :].to(device))
    features = features.detach().cpu().numpy().reshape(features.shape[0], -1)

    features_2d = tsne_on_pca(features, is_PCA=False)
    features_2d = features_2d.reshape(nr_mv, nr_char, -1)

    plt.figure(figsize=(7, 4))
    colors = cm.rainbow(np.linspace(0, 1, nr_char))
    for i in range(nr_char):
        x = features_2d[:, i, 0]
        y = features_2d[:, i, 1]
        plt.scatter(x, y, c=colors[i], label=characters[i])

    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.tight_layout(rect=[0,0,0.75,1])
    plt.savefig(save_path)


def cluster_view(net, cluster_data, device, save_path):
    data, views = cluster_data[0], cluster_data[3]
    idx = np.random.randint(data.shape[1] - 1)  # np.linspace(0, data.shape[1] - 1, 4, dtype=int).tolist()
    data = data[:, idx, :, :, :]

    nr_mc, nr_view = data.shape[0], data.shape[1]
    labels = np.arange(0, nr_view).reshape(1, -1)
    labels = np.tile(labels, (nr_mc, 1)).reshape(-1)
    
    if hasattr(net, 'static_encoder'):
        features = net.static_encoder(data.contiguous().view(-1, data.shape[2], data.shape[3])[:, :-2, :].to(device))
    else:
        features = net.view_encoder(data.contiguous().view(-1, data.shape[2], data.shape[3])[:, :-2, :].to(device))
    features = features.detach().cpu().numpy().reshape(features.shape[0], -1)

    features_2d = tsne_on_pca(features, is_PCA=False)
    features_2d = features_2d.reshape(nr_mc, nr_view, -1)

    plt.figure(figsize=(7, 4))
    colors = cm.rainbow(np.linspace(0, 1, nr_view))
    for i in range(nr_view):
        x = features_2d[:, i, 0]
        y = features_2d[:, i, 1]
        plt.scatter(x, y, c=colors[i], label=views[i])

    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.savefig(save_path)


def cluster_motion(net, cluster_data, device, save_path, nr_anims=15, mode='both'):
    data, animations = cluster_data[0], cluster_data[1]
    idx = np.linspace(0, data.shape[0] - 1, nr_anims, dtype=int).tolist()
    data = data[idx]
    animations = animations[idx]
    if mode == 'body':
        data = data[:, :, 0, :, :].reshape(nr_anims, -1, data.shape[3], data.shape[4])
    elif mode == 'view':
        data = data[:, 3, :, :, :].reshape(nr_anims, -1, data.shape[3], data.shape[4])
    else:
        data = data[:, :4, ::2, :, :].reshape(nr_anims, -1, data.shape[3], data.shape[4])

    nr_anims, nr_cv = data.shape[:2]
    labels = np.arange(0, nr_anims).reshape(-1, 1)
    labels = np.tile(labels, (1, nr_cv)).reshape(-1)
    
    features = net.mot_encoder(data.contiguous().view(-1, data.shape[2], data.shape[3]).to(device))
    features = features.detach().cpu().numpy().reshape(features.shape[0], -1)

    features_2d = tsne_on_pca(features)
    features_2d = features_2d.reshape(nr_anims, nr_cv, -1)
    if features_2d.shape[1] < 5:
        features_2d = np.tile(features_2d, (1, 2, 1))

    plt.figure(figsize=(8, 4))
    colors = cm.rainbow(np.linspace(0, 1, nr_anims))
    for i in range(nr_anims):
        x = features_2d[i, :, 0]
        y = features_2d[i, :, 1]
        plt.scatter(x, y, c=colors[i], label=animations[i])

    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.tight_layout(rect=[0,0,0.8,1])
    plt.savefig(save_path)


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, choices=['skeleton', 'view', 'full'], required=True,
                        help='which structure to use.')
    parser.add_argument('-p', '--model_path', type=str, default="model/pretrained_view.pth")
    parser.add_argument('--phase', type=str, default="test", choices=['train', 'test'])
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False, help="specify gpu ids")
    args = parser.parse_args()

    # set config
    config.initialize(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load trained model
    net = get_autoencoder(config)
    net.load_state_dict(torch.load(args.model_path))
    net.to(config.device)
    net.eval()
    
    # get dataset
    train_ds = MixamoDatasetForFull(args.phase, config)
    cluster_data = train_ds.get_cluster_data()

    # score, img = cluster_body(net, cluster_data, device, './cluster_body.png')
    if args.name == 'view':
        cluster_view(net, cluster_data, device, './cluster_view.png')
        cluster_motion(net, cluster_data, device, './cluster_motion.png')
    elif args.name == 'skeleton':
        cluster_body(net, cluster_data, device, './cluster_body.png')
        cluster_motion(net, cluster_data, device, './cluster_motion.png', mode='body')
    else: 
        cluster_motion(net, cluster_data, device, './cluster_motion.png')


if __name__ == '__main__':
    test()

