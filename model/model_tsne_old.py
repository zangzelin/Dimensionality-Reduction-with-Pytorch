import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn import manifold, datasets
from torch.autograd import Variable
import numpy as np


class TSNE(nn.Module):
    def __init__(self, data, device, args, n_dim=2,):
        self.decive = device
        self.n_points = data.shape[0]
        self.n_dim = n_dim
        super(TSNE, self).__init__()
        self.data = torch.tensor(data)
        self.perplexity = args.perplexity

        self.pij = self.CalPij(self.data).float().to(self.decive)
        from sklearn.decomposition import PCA
        clf = PCA(n_components=2)
        clf.fit(data.detach().cpu().numpy())
        e = clf.transform(data.detach().cpu().numpy())
        self.pij[self.pij < 1e-16] = 1e-16
        self.output = torch.nn.Parameter(torch.tensor(e.astype(np.float32)))

    def Distance_squared(self, data1, data2, ):
        x = data1
        y = data2
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        d = dist.clamp(min=1e-36)
        return d

    def GetEmbedding(self, ):
        return self.output.detach().cpu().numpy()

    def CalPij(self, X, perplexity=90.0):
        perplexity = self.perplexity
        dis_squared = pairwise_distances(
            X.detach().cpu().numpy(),  metric='euclidean', squared=True)

        pij = manifold._t_sne._joint_probabilities(
            dis_squared, perplexity, False)
        return torch.tensor(squareform(pij))

    def forward(self, sample_index_i, sample_index_j):

        out1 = self.output
        # out2 = self.output[sample_index_j]
        dis = self.Distance_squared(out1, out1)

        qij_top = 1/(1+dis)
        sum_m = (
            qij_top.sum() -
            qij_top[torch.eye(dis.shape[0]) == 1].sum()
        )
        qij = qij_top / sum_m
        qij = torch.max(qij, torch.tensor([1e-36], device=self.decive))

        pij = self.pij
        loss_kld = pij * (torch.log(pij) - torch.log(qij))
        return loss_kld.sum()

    def __call__(self, *args):
        return self.forward(*args)
