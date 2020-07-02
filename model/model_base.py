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

    def forward(self, input_data):

        # input_1 = self.data.float()[sample_index_i]

        input_c = input_data
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

    def Loss(self, latent, input_data):

        # input_data = self.data[sample_index_i]
        # latent =

        loss_mae_distance = self.MorphicLossItem(
            input_data, latent
        )
        return loss_mae_distance

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
