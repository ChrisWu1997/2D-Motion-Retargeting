import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.switch_backend('agg')
import torch
from dataset import get_dataloaders
import cv2
import time

def tsne_on_pca(arr):
    """
    visualize through t-sne on pca reduced data
    :param arr: (nr_examples, nr_features)
    :return:
    """
    pca_50 = PCA(n_components=50)
    res = pca_50.fit_transform(arr)
    tsne_2 = TSNE(n_components=2)
    res = tsne_2.fit_transform(res)
    return res


def cluster_body(net, cluster_data, device, save_path, is_draw=True):
    data, characters = cluster_data[0], cluster_data[1]
    nr_chars, nr_anims = data.shape[:2]
    labels = np.arange(0, nr_chars).reshape(-1, 1)
    labels = np.tile(labels, (1, nr_anims)).reshape(-1)

    data = data[:, :, :-2, :]
    features = net.body_encoder(data.view(-1, data.shape[2], data.shape[3]).to(device))
    features = features.detach().cpu().numpy().reshape(features.shape[0], -1)

    sil_score = silhouette_score(features, labels)

    if not is_draw:
        return sil_score, None

    features_2d = tsne_on_pca(features)
    features_2d = features_2d.reshape(nr_chars, nr_anims, -1)

    plt.figure(figsize=(7, 4))
    colors = cm.rainbow(np.linspace(0, 1, nr_chars))
    for i in range(nr_chars):
        x = features_2d[i, :, 0]
        y = features_2d[i, :, 1]
        plt.scatter(x, y, c=colors[i], label=characters[i])

    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.tight_layout(rect=[0,0,0.75,1])
    plt.savefig(save_path)

    img = cv2.imread(save_path)

    return sil_score, img


def cluster_motion(net, cluster_data, device, save_path, nr_anims=15, is_draw=True):
    data, animations = cluster_data[0], cluster_data[2]
    data = data[:, :nr_anims, :, :]
    nr_chars, nr_anims = data.shape[:2]
    labels = np.arange(0, nr_anims).reshape(1, -1)
    labels = np.tile(labels, (nr_chars, 1)).reshape(-1)

    features = net.mot_encoder(data.contiguous().view(-1, data.shape[2], data.shape[3]).to(device))
    features = features.detach().cpu().numpy().reshape(features.shape[0], -1)

    sil_score = silhouette_score(features, labels)

    if not is_draw:
        return sil_score, None

    features_2d = tsne_on_pca(features)
    features_2d = features_2d.reshape(nr_chars, nr_anims, -1)

    plt.figure(figsize=(7, 4))
    colors = cm.rainbow(np.linspace(0, 1, nr_anims))
    for i in range(nr_anims):
        x = features_2d[:, i, 0]
        y = features_2d[:, i, 1]
        plt.scatter(x, y, c=colors[i], label=animations[i])

    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.tight_layout(rect=[0,0,0.6,1])
    plt.savefig(save_path)

    img = cv2.imread(save_path)

    return sil_score, img


def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path = '/home1/wurundi/code/fbxMotionDisentangled/2d.v1fix.aug.fbxjoints.11anim.15joints.layer64/train_log/model/epoch300.pth.tar'
    net = torch.load(path)['net']

    train_ds = get_dataloaders('train', batch_size=1)

    cluster_data = train_ds.dataset.get_cluster_data()

    since = time.time()
    score, img = cluster_body(net, cluster_data, device, './cluster_body.png')
    print('times: {}'.format(time.time() - since))
    print(score)

    since = time.time()
    score, img = cluster_motion(net, cluster_data, device, './cluster_motion.png')
    print('times: {}'.format(time.time() - since))
    print(score)


if __name__ == '__main__':
    test()
