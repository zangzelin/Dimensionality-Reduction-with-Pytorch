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
    def __init__(self, data, device, n_dim=2,):
        self.decive = device
        self.n_points = data.shape[0]
        self.n_dim = n_dim
        super(TSNE, self).__init__()
        self.data = torch.tensor(data)

        self.pij = self.CalPij(self.data).float().to(self.decive)
        # self.pij = self.x2p_torch(self.data).to(self.decive)
        self.pij[self.pij < 1e-16] = 1e-16
        self.output = torch.nn.Parameter(torch.randn(self.n_points, n_dim))

    def Distance_squared(self, data):
        x = data
        y = data
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        d = dist.clamp(min=1e-8)
        return d

    def GetEmbedding(self, ):
        return self.output.detach().cpu().numpy()

    def Hbeta_torch(self, D, beta=1.0):
        P = torch.exp(-D.clone() * beta)
        sumP = torch.sum(P)
        H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
        P = P / sumP
        return H, P

    def CalPi_j(self, X, perplexity=30.0):
        dis = self.Distance_squared(X)

        pi_j_top = torch.exp(-1*dis/2/perplexity/perplexity)
        sum_m = (pi_j_top.sum(dim=1) -
                 pi_j_top[torch.eye(dis.shape[0]).bool()]).expand_as(dis).t()
        pi_j = torch.div(pi_j_top, sum_m)
        return pi_j

    def CalPj_i(self, X, perplexity=30.0):
        dis = self.Distance_squared(X)

        pj_i_top = torch.exp(-1*dis/2/perplexity/perplexity)
        sum_m = (pj_i_top.sum(dim=1) -
                 pj_i_top[torch.eye(dis.shape[0]).bool()]).expand_as(dis)
        pj_i = torch.div(pj_i_top, sum_m)
        return pj_i

    def CalPij(self, X, perplexity=90.0):
        dis_squared = pairwise_distances(X.detach().cpu().numpy(),  metric='euclidean', squared=True)

        pij = manifold._t_sne._joint_probabilities(
            dis_squared, perplexity, False)
        return torch.tensor(squareform(pij))

    def x2p_torch(self, X, tol=1e-5, perplexity=30.0):
        """
            Performs a binary search to get P-values in such a way that each
            conditional Gaussian has the same perplexity.
        """

        # Initialize some variables
        # print("Computing pairwise distances...")
        (n, d) = X.shape

        sum_X = torch.sum(X*X, 1)
        D = torch.add(torch.add(-2 * torch.mm(X, X.t()), sum_X).t(), sum_X)

        P = torch.zeros(n, n)
        beta = torch.ones(n, 1)
        logU = torch.log(torch.tensor([perplexity]))
        n_list = [i for i in range(n)]

        # Loop over all datapoints
        for i in range(n):

            # Print progress
            if i % 500 == 0:
                print("Computing P-values for point %d of %d..." % (i, n))

            # Compute the Gaussian kernel and entropy for the current precision
            # there may be something wrong with this setting None
            betamin = None
            betamax = None
            Di = D[i, n_list[0:i]+n_list[i+1:n]]

            (H, thisP) = self.Hbeta_torch(Di, beta[i])

            # Evaluate whether the perplexity is within tolerance
            Hdiff = H - logU
            tries = 0
            while torch.abs(Hdiff) > tol and tries < 50:

                # If not, increase or decrease precision
                if Hdiff > 0:
                    betamin = beta[i].clone()
                    if betamax is None:
                        beta[i] = beta[i] * 2.
                    else:
                        beta[i] = (beta[i] + betamax) / 2.
                else:
                    betamax = beta[i].clone()
                    if betamin is None:
                        beta[i] = beta[i] / 2.
                    else:
                        beta[i] = (beta[i] + betamin) / 2.

                # Recompute the values
                (H, thisP) = self.Hbeta_torch(Di, beta[i])

                Hdiff = H - logU
                tries += 1

            # Set the final row of P
            P[i, n_list[0:i]+n_list[i+1:n]] = thisP

        # Return final P-matrix
        return P

    def JointP(self, Y):
        n = Y.shape[0]
        sum_Y = torch.sum(Y*Y, 1)
        num = -2. * torch.mm(Y, Y.t())
        num = 1. / (1. + torch.add(torch.add(num, sum_Y).t(), sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / torch.sum(num)
        Q = torch.max(Q, torch.tensor([1e-12]))
        return Q


    def forward(self):
        dis = self.Distance_squared(self.output)

        n_diagonal = dis.size()[0]
        part = (1 + dis).pow(-1.0).sum() - n_diagonal

        qij_top = 1/(1+dis)

        sum_m = (
            qij_top.sum() -
            qij_top[torch.eye(dis.shape[0]).bool()].sum()
        )
        qij = qij_top / sum_m

        qij = torch.max(qij, torch.tensor([1e-36], device=self.decive))
        loss_kld = self.pij * (torch.log(self.pij) - torch.log(qij))
        return loss_kld.sum()

    def __call__(self, *args):
        return self.forward(*args)
