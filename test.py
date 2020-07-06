import sklearn
import dataloader
import torch
import matplotlib.pyplot as plt
import numpy as np
import umap

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_umap(data_train, label_train):
    #     data_train, data_test, label_train, label_test = dataloader.GetData(
    #         data_name='mnist', device=device)
    print('into')
    # clf = sklearn.manifold.TSNE(n_components=3, n_iter=250)
    clf = umap.UMAP(n_components=3, n_epochs=250, n_neighbors=3)
    em = clf.fit_transform(data_train.detach().cpu())
    #         em = Model.GetEmbedding()
    # plt.scatter(em[:, 0], em[:, 1], c=label_train.detach().cpu(), s=1)
    # plt.savefig('tsne_sklearn.png')
    np.savetxt('umap_input.txt', data_train.detach().cpu().numpy())
    np.savetxt('umap_latent.txt', em)
    # plt.close()


def test_sklearn(data_train, label_train):
    #     data_train, data_test, label_train, label_test = dataloader.GetData(
    #         data_name='mnist', device=device)
    print('into')
    clf = sklearn.manifold.TSNE(n_components=3, n_iter=250)
    # clf = umap.UMAP(n_components=3)
    em = clf.fit_transform(data_train.detach().cpu())
    #         em = Model.GetEmbedding()
    # plt.scatter(em[:, 0], em[:, 1], c=label_train.detach().cpu(), s=1)
    # plt.savefig('tsne_sklearn.png')
    np.savetxt('sklearn_input.txt', data_train.detach().cpu().numpy())
    np.savetxt('sklearn_latent.txt', em)
