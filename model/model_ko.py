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


class MAE_MLP(nn.Module):

    def __init__(self, data, device, args, n_dim=2,):
        self.device = device
        super().__init__()
        # self.input_size = input_size
        self.args = args
        self.NetworkStructure = args.NetworkStructure
        self.index_latent = (len(args.NetworkStructure)-1)*2 - 1

        self.plot_index_list = [0]
        self.network = nn.ModuleList()
        for i in range(len(self.NetworkStructure)-1):
            self.network.append(
                nn.Linear(
                    self.NetworkStructure[i], self.NetworkStructure[i+1])
            )
            if i != len(self.NetworkStructure)-2:
                self.network.append(
                    nn.LeakyReLU(0.1)
                )

        self.dis_input = self.Distance_squared(data, data)
        self.data = data

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

    def forward(self, sample_index_i):

        input_1 = self.data.float()[sample_index_i]

        input_c = input_1
        for i, layer in enumerate(self.network):
            output_c = layer(input_c)
            input_c = output_c

        return output_c

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

    def KO(self, latent, input_data):

        d_data1, indices_rank_1, rank_indices_1 = self.kNNGraph_order(latent)
        d_data2, indices_rank_2, rank_indices_2 = self.kNNGraph_order(
            input_data)
        d_data1, kNN_mask_data1 = self.kNNGraph(latent)
        d_data2, kNN_mask_data2 = self.kNNGraph(input_data)

        # d_data1_n = d_data1
        # d_data2_n = d_data2
        # cc = torch.zeros_like(d_data1)
        # for i in range(d_data1.shape[0]):
        #     for j in range(d_data1.shape[1]):
        #         # ind_1 = indices_data1[i, j]
        #         ind_2 = indices_rank_2[i, j]
        #         for k in range(indices_rank_1.shape[0]):
        #             if k == ind_2:
        #                 break
        #         # k = indices_data1[i].indices()
        #         cc[i, j] = torch.abs(d_data1[i, j] - d_data1[i, k])
        # loss += 0
        # d_data1
        b = d_data1
        c = torch.zeros_like(d_data1)
        for i in range(d_data1.shape[0]):
            c[i] = d_data1[i][indices_rank_2[i]]
        l = torch.abs(b-c)[kNN_mask_data2+kNN_mask_data2 != 0]
        print('-------------------------------->', torch.sum(kNN_mask_data2))
        loss = l.sum()

        a = torch.max(1 - d_data1, torch.zeros_like(d_data1)).sum() * 100

        print(loss, a)

        return (loss + a)/100000

    def Loss(self, latent, sample_index_i):

        input_data = self.dis_input[sample_index_i]
        # latent =
        loss = self.KO(latent, input_data)

        return [loss]

    def RankLoss(self, data1, data2, d_data1, d_data2, kNN_mask_data1, kNN_mask_data2):

        data_not_near = d_data2[kNN_mask_data1 == 0]
        data_not_near = torch.min(
            data_not_near, torch.ones_like(data_not_near)*10)

        return torch.mean(d_data2)*0.00001-torch.mean(data_not_near)

    def kNNGraph(self, data):

        # import time
        k = self.args.perplexity
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
        # cut = s_[:, k+1].view(-1, 1).expand_as(d)
        # print(s_[:, k+1].view(-1,1).shape)
        # kNN_mask[d < cut] = 1
        indices = indices[:, :k+1]
        for i in range(kNN_mask.size(0)):
            kNN_mask[i, :][indices[i]] = 1
        kNN_mask[torch.eye(kNN_mask.shape[0], dtype=bool)] = 0
        # kNN_mask = torch.tensor(kNN_mask).to(device)
        # self.neighbor = indices
        return d, kNN_mask.bool()

    def MorphicLossItem(self, data1, data2):

        # print(data2)
        d_data1, kNN_mask_data1 = self.kNNGraph(data1)
        d_data2, kNN_mask_data2 = self.kNNGraph(data2)

        loss_mae_distance = self.DistanceLoss(
            data1, data2, d_data1, d_data2, kNN_mask_data1, kNN_mask_data2)
        rankloss = self.RankLoss(
            data1, data2, d_data1, d_data2, kNN_mask_data1, kNN_mask_data2)
        # print(loss_mae_d)
        return loss_mae_distance+rankloss

    def kNNGraph_order(self, data):

        # import time
        k = self.args.perplexity
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
        s_, indices_rank = torch.sort(d, dim=1)
        rank_indices = torch.sort(indices_rank, dim=1)
        # self.indices = indices
        # # cut = s_[:, k+1].view(-1, 1).expand_as(d)
        # # print(s_[:, k+1].view(-1,1).shape)
        # # kNN_mask[d < cut] = 1
        # indices = indices[:, :k+1]
        # for i in range(kNN_mask.size(0)):
        #     kNN_mask[i, :][indices[i]] = 1
        # kNN_mask[torch.eye(kNN_mask.shape[0], dtype=bool)] = 0
        # kNN_mask = torch.tensor(kNN_mask).to(device)
        # self.neighbor = indices
        return d, indices_rank, rank_indices

    def DistanceLoss(self, data1, data2, d_data1, d_data2,
                     kNN_mask_data1, kNN_mask_data2):

        norml_data1 = torch.sqrt(torch.tensor(float(data1.shape[1])))
        norml_data2 = torch.sqrt(torch.tensor(float(data2.shape[1])))

        mask_u = kNN_mask_data1
        D1_1 = (d_data1/norml_data1)[mask_u]
        D1_2 = (d_data2/norml_data2)[mask_u]
        Error1 = (D1_1 - D1_2) / 1
        loss2_1 = torch.norm(Error1)/torch.sum(mask_u)

        loss_mae_distance = 1*loss2_1

        return loss_mae_distance  # , loss_mae_mutex
