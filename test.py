import sklearn
import dataloader
import torch
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_train, data_test, label_train, label_test = dataloader.GetData(
        data_name='mnist', device = device)
clf = sklearn.manifold.TSNE()
em = clf.fit_transform(data_train.detach().cpu())
#         em = Model.GetEmbedding()
plt.scatter(em[:, 0], em[:, 1], c=label_train.detach().cpu())
plt.savefig('lll2.png')
plt.close()