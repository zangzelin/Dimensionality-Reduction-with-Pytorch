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
        # input(self.pij)
        from sklearn.decomposition import PCA
        clf = PCA(n_components=2)
        clf.fit(data.detach().cpu().numpy())
        e = clf.transform(data.detach().cpu().numpy())
        # self.pij = self.x2p_torch(self.data).to(self.decive)
        self.pij[self.pij < 1e-16] = 1e-16
        self.output = torch.nn.Parameter(torch.tensor(e.astype(np.float32)))

    def CalPij(self, X, perplexity=90.0):
        perplexity = self.perplexity
        dis_squared = pairwise_distances(
            X.detach().cpu().numpy(),  metric='euclidean', squared=True)

        pij = manifold.t_sne._joint_probabilities(
            dis_squared, perplexity, False)
        return torch.tensor(squareform(pij))

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

    def Test(self, data):

        # input_1 = self.data.float()[sample_index_i]

        input_c = data.float()
        for i, layer in enumerate(self.network):
            output_c = layer(input_c)
            input_c = output_c

        return output_c.detach().cpu().numpy()

    def G_forward(self, D2data):
        output_info = []
        input_c = D2data
        for i, layer in enumerate(self.network):
            if i > (len(self.NetworkStructure)-2)*2:
                output_c = layer(input_c)
                output_info.append(output_c)
                if self.index_latent-1 == i:
                    input_c = output_c
                else:
                    input_c = output_c

        return output_c

    def Loss(self, latent, input_data):

        # input_data = self.data[sample_index_i]
        # latent =

        dis = self.Distance_squared(latent, latent)

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

    def RankLoss(self, data1, data2, d_data1, d_data2, kNN_mask_data1, kNN_mask_data2):

        # no_near_dis = d_data2
        # no_near_dis[kNN_mask_data1 == 0] = 0
        # # import pdb
        # near_dis = torch.clone(d_data2)
        # near_dis[kNN_mask_data1 == 1] = 0
        # near_dis_max, _ = near_dis.max(dim=0)

        # # pdb.set_trace()
        # near_dis_max = near_dis_max.view(1, -1).expand_as(kNN_mask_data1)
        # no_near_dis = torch.min(near_dis_max*3, no_near_dis)

        # near_dis = torch.clone(d_data2)
        # near_dis[kNN_mask_data1 == 0] = 0
        # near_dis_max, _ = near_dis.max(dim=0)

        data_not_near = d_data2[kNN_mask_data1 == 0]
        data_not_near = torch.min(
            data_not_near, torch.ones_like(data_not_near)*10)

        return 0-torch.mean(data_not_near)

    def MorphicLossItem(self, input_data, latent):

        # print(data2)
        d_data1, kNN_mask_data1 = self.kNNGraph(input_data)
        d_data2, kNN_mask_data2 = self.kNNGraph(latent)

        d_data1_max = d_data1.max()
        d_data1 = d_data1/d_data1_max
        # d_data1_max,_ = d_data1.max(dim=1)
        # d_data1 = d_data1/d_data1_max

        d_data1 = torch.exp(1+d_data1*5)

        self.Graph_input = kNN_mask_data1.detach().cpu().numpy()
        self.Graph_link = d_data1.detach().cpu().numpy()

        loss_mae_distance = self.DistanceLoss(
            input_data, latent, d_data1, d_data2, kNN_mask_data1, kNN_mask_data2)

        d_data2 = torch.log(1+d_data2)
        huchi = d_data2.mean()
        # rankloss = self.RankLoss(
        #     data1, data2, d_data1, d_data2, kNN_mask_data1, kNN_mask_data2)
        # print(loss_mae_d)
        return loss_mae_distance, -1*huchi/3  # +rankloss

    def GetInput(self,):
        return self.Graph_input

    def Getlink(self,):
        return self.Graph_link

    def kNNGraph(self, data):

        # import time
        k = self.args.perplexity
        # print(k)
        # input(k)
        batch_size = data.shape[0]

        x = data.to(self.device)
        y = data.to(self.device)
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        d = dist.clamp(min=1e-8).sqrt()  # for numerical stabili

        kNN_mask = torch.zeros((batch_size, batch_size,), device=self.device)
        s_, indices = torch.sort(d, dim=1)
        self.indices = indices
        cut = s_[:, k+1].view(-1, 1).expand_as(d)
        # print(s_[:, k+1].view(-1, 1).shape)
        kNN_mask[d < cut] = 1
        # indices = indices[:, :k+1]
        # for i in range(kNN_mask.size(0)):
        #     kNN_mask[i, :][indices[i]] = 1
        # kNN_mask[torch.eye(kNN_mask.shape[0], dtype=bool)] = 0
        # kNN_mask = torch.tensor(kNN_mask).to(device)
        # self.neighbor = indices
        return d, kNN_mask.bool()

    def DistanceLoss(self, data1, data2, d_data1, d_data2,
                     kNN_mask_data1, kNN_mask_data2):

        # d_data1 = torch.exp(1.1+d_data1)
        # d_data2 = torch.exp(1.1+d_data2)

        norml_data1 = torch.sqrt(torch.tensor(float(data1.shape[1])))
        norml_data2 = torch.sqrt(torch.tensor(float(data2.shape[1])))

        mask_u = (kNN_mask_data1 > 0)
        D1_1 = (d_data1/norml_data1)[mask_u]
        D1_2 = (d_data2/norml_data2)[mask_u]
        Error1 = (D1_1 - D1_2) / 1
        loss2_1 = (torch.norm(Error1))/torch.sum(mask_u)

        loss_mae_distance = 1*loss2_1

        return loss_mae_distance  # , loss_mae_mutex

    def forward(self, input_data):

        input_c = input_data
        for i, layer in enumerate(self.network):
            output_c = layer(input_c)
            input_c = output_c

        return output_c

    def __call__(self, *args):
        return self.forward(*args)
