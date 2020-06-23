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


class TSNE_NN(nn.Module):
    def __init__(self, data, device, args, n_dim=2,):
        self.device = device
        self.n_points = data.shape[0]
        self.n_dim = n_dim
        super(TSNE_NN, self).__init__()
        self.data = torch.tensor(data)
        self.args = args
        self.perplexity = args.perplexity

        self.pij = self.CalPij(self.data).float().to(self.device)
        # input(self.pij)
        # self.pij = self.x2p_torch(self.data).to(self.device)
        self.pij[self.pij < 1e-36] = 1e-36
        self.r = 0
        # self.output = torch.nn.Parameter(torch.randn(self.n_points, n_dim))
        self.NetworkStructure = [64, 5000, 2]
        self.network = nn.ModuleList()
        for i in range(len(self.NetworkStructure)-1):
            self.network.append(
                nn.Linear(
                    self.NetworkStructure[i], self.NetworkStructure[i+1])
            )
            if i != len(self.NetworkStructure)-2:
                self.network.append(
                    nn.Sigmoid()
                )
        # self.r_pl = 1
        # self.r_kl = 1
        self.k = args.perplexity
        self.near_input = self.kNNGraphNear(self.dis)
        # input(self.network)

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

    def GetEmbedding(self):

        input_c_1 = self.data.float()
        for i, layer in enumerate(self.network):
            output_c_1 = layer(input_c_1)
            input_c_1 = output_c_1
        return output_c_1.detach().cpu().numpy()

    def CalPij(self, X, perplexity=30.0):
        perplexity = self.perplexity

        dis_squared = pairwise_distances(
            X.detach().cpu().numpy(),  metric='euclidean',
            squared=True, n_jobs=-1)
        self.dis = torch.tensor(dis_squared.astype(np.float32))
        pij = manifold.t_sne._joint_probabilities(
            dis_squared, perplexity, False)
        return torch.tensor(squareform(pij))

    def CossimiSlow(self, data):

        # print(self.out)
        eps = 1e-8
        a_n, b_n = data.norm(dim=1)[:, None], data.norm(dim=1)[:, None]
        a_norm = data / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = data / torch.max(b_n, eps * torch.ones_like(b_n))
        out = torch.mm(a_norm, b_norm.transpose(0, 1))[
            torch.eye(a_norm.shape[0]) == 0
        ]

        return out.abs().mean()

    def KLLoss(self, dis, sample_index_i):
        qij_top = 1/(1+dis)
        sum_m = (
            qij_top.sum() -
            qij_top[torch.eye(dis.shape[0]) == 1].sum()
        )
        qij = qij_top / sum_m
        qij = torch.max(qij, torch.tensor([1e-36], device=self.device))
        pij = self.pij[sample_index_i][:, sample_index_i]

        loss_kld = pij * (torch.log(pij) - torch.log(qij))

        return loss_kld.sum()

    def kNNGraphNear(self, dis):

        k = self.k

        kNN_mask = torch.zeros(dis.shape, device=self.device)
        s_, indices = torch.sort(dis, dim=1)
        self.indices = indices
        indices = indices[:, :k+1]
        for i in range(kNN_mask.size(0)):
            kNN_mask[i, :][indices[i]] = 1
        kNN_mask[torch.eye(kNN_mask.shape[0], dtype=bool)] = 0

        return kNN_mask

    def PLLoss(self, dis, sample_index_i):
        D2_2 = (dis)[self.near_input[sample_index_i]
                     [:, sample_index_i] == False]
        Error2 = D2_2
        Error2[Error2 > 10] = 10
        loss2_2 = torch.norm(Error2) / \
            torch.sum(self.near_input[sample_index_i]
                      [:, sample_index_i] == False)
        return -500*loss2_2

    def Test(self, testdata):

        input_c_1 = testdata.float()
        for i, layer in enumerate(self.network):
            output_c_1 = layer(input_c_1)
            input_c_1 = output_c_1
            try:
                weigh_list.append(layer.weight)
            except:
                pass
        return output_c_1.detach().cpu().numpy()

    def Loss(self, output_c_1, sample_index_i):

        dis = self.Distance_squared(output_c_1, output_c_1)
        KL_loss = self.KLLoss(dis, sample_index_i)
        PL_loss = self.PLLoss(dis, sample_index_i)
        # print(KL_loss.item(), PL_loss.item())
        return [
            self.args.rate_plloss*PL_loss,
            self.args.rate_klloss*KL_loss
        ]

    def forward(self, sample_index_i):

        input_1 = self.data.float()[sample_index_i]

        weigh_list = []
        input_c_1 = input_1
        for i, layer in enumerate(self.network):
            output_c_1 = layer(input_c_1)
            input_c_1 = output_c_1
            try:
                weigh_list.append(layer.weight)
            except:
                pass

        return self.Loss(output_c_1, sample_index_i)  # + loss_pl*self.r

    def __call__(self, *args):
        return self.forward(*args)
