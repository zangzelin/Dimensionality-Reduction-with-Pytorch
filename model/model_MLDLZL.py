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


class MLDLZL_MLP(nn.Module):

    def __init__(self, data, device, args, n_dim=2,):
        
        self.device = device
        super().__init__()
        self.args = args
        self.NetworkStructure = args.NetworkStructure

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

    def forward(self, input_data):

        input_c = input_data
        for i, layer in enumerate(self.network):
            output_c = layer(input_c)
            input_c = output_c

        return output_c

    def Test(self, data):

        input_c = data.float()
        for i, layer in enumerate(self.network):
            output_c = layer(input_c)
            input_c = output_c

        return output_c.detach().cpu().numpy()

    def Loss(self, latent, input_data):

        loss_mae_distance = self.DistanceAndPushAwayLoss(
            input_data, latent
        )
        return loss_mae_distance

    def DistanceAndPushAwayLoss(self, input_data, latent):

        # print(data2)
        distanceInput, knnInput = self.kNNGraph(input_data)
        distanceLatent, knnLatent = self.kNNGraph(latent)

        distanceInput_max = distanceInput.max()
        distanceInput = distanceInput/distanceInput_max

        distanceInput = torch.exp(1+distanceInput*5)

        self.Graph_input = knnInput.detach().cpu().numpy()
        self.Graph_link = distanceInput.detach().cpu().numpy()

        distanceLoss = self.DistanceLoss(
            input_data, latent,
            distanceInput, distanceLatent,
            knnInput, knnLatent)

        pushAwayLoss = torch.log(1+distanceLatent).mean()

        return distanceLoss, -1*pushAwayLoss/3  # +rankloss

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
        kNN_mask[d < cut] = 1

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
