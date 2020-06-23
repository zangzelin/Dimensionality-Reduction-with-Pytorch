
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
# from matplotlib.patches import Ellipse
# from scipy.spatial.distance import squareform
# from sklearn import datasets, manifold
# from sklearn.metrics.pairwise import pairwise_distances

import dataloader
import model.model_tsne_nn as model_tsne_nn
import model.model_tsne as model_tsne
import tool

matplotlib.use('Agg')


def train(args, Model, Loss, device, data, target, optimizer, epoch):

    BATCH_SIZE = args.batch_size

    # batch_idx = 0
    Model.train()
    data, target = data.to(device), target.to(device)

    num_train_sample = data.shape[0]
    num_batch = (num_train_sample-0.5)//BATCH_SIZE + 1
    rand_index_i = torch.randperm(num_train_sample)
    # rand_index_j = torch.randperm(num_train_sample)
    train_loss_sum = [0, 0, 0, 0, 0, 0, 0]

    for batch_idx in torch.arange(0, num_batch):
        start = (batch_idx * BATCH_SIZE).int()
        end = torch.min(
            torch.tensor(
                [batch_idx * BATCH_SIZE + BATCH_SIZE, num_train_sample]
            )
        )
        sample_index_i = rand_index_i[start: end.int()]
        # sample_index_j = rand_index_j[start: end.int()]

        optimizer.zero_grad()

        loss_list = Model(sample_index_i)
        for i, loss_item in enumerate(loss_list):
            loss_item.backward(retain_graph=True)
            train_loss_sum[i] += loss_item.item()

        # train_loss_sum.backward()
        optimizer.step()

    loss_list = [loss_list[i].item() for i in range(len(loss_list))]
    print('Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {}'.format(
        epoch, batch_idx * len(data), num_train_sample,
        BATCH_SIZE*100.*batch_idx / num_train_sample,
        loss_list
    ))
    return loss_list


def test(args, Model, Loss, device, data, target, optimizer, epoch):

    BATCH_SIZE = args.batch_size

    # batch_idx = 0
    # Model.train()
    data, target = data.to(device), target.to(device)

    num_train_sample = data.shape[0]

    em = Model.Test(data)

    return em


def GetParam():

    parser = argparse.ArgumentParser(description='zelin zang author')
    parser.add_argument('--name', type=str, default='zzl',)

    # data set param
    parser.add_argument('--method', type=str, default='tsne_nn',)
    parser.add_argument('--data_name', type=str, default='mnist',)
    parser.add_argument('--data_trai_n', type=int, default=10000,)
    parser.add_argument('--data_test_n', type=int, default=10000,)

    # model param
    parser.add_argument('--perplexity', type=int, default=30,)
    parser.add_argument('--rate_plloss', type=float, default=1,)
    parser.add_argument('--rate_klloss', type=float, default=1,)

    # train param
    parser.add_argument('--batch_size', type=int, default=10000,)
    parser.add_argument('--epochs', type=int, default=10000)
    # parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',)
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',)
    parser.add_argument('--no-cuda', action='store_true', default=False,)
    parser.add_argument('--seed', type=int, default=1, metavar='S',)
    parser.add_argument('--log_interval', type=int, default=100,)
    args = parser.parse_args()

    args.batch_size = min(args.batch_size, args.data_trai_n, args.data_test_n,)

    return args


def main():

    args = GetParam()
    path = tool.GetPath(args.name)
    tool.SaveParam(path, args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    data_train, data_test, label_train, label_test = dataloader.GetData(
        args, device=device, pca=64)
    tool.SetSeed(args.seed)
    gif1 = tool.GIFPloter()
    gif2 = tool.GIFPloter()

    if args.method == 'tsne_nn':
        Model = model_tsne_nn.TSNE_NN(
            data_train, device=device, args=args).to(device)
    elif args.method == 'tsne':
        Model = model_tsne.TSNE(
            data_train, device=device, args=args).to(device)
    Loss = None  # model.TSNE_LOSS().to(device)
    optimizer = optim.Adam(Model.parameters(), lr=args.lr)

    loss_his = []
    for epoch in range(1, args.epochs + 1):
        loss_item = train(args, Model, Loss, device, data_train,
                          label_train, optimizer, epoch)

        loss_his.append(loss_item)

        args.rate_plloss = max(1-epoch/1000, 0)
        if epoch % args.log_interval == 0:
            if epoch > 2000:
                args.lr = tool.LearningRateScheduler(
                    loss_his[-1000:], optimizer, args.lr)

            em = test(args, Model, Loss, device, data_test,
                      label_test, optimizer, epoch)

            gif1.AddNewFig(em,
                           label_test.detach().cpu(),
                           loss_his, path=path,
                           title_='test_epoch{}.png'.format(epoch))
            gif2.AddNewFig(Model.GetEmbedding(),
                           label_train.detach().cpu(),
                           loss_his, path=path,
                           title_='epoch{}.png'.format(epoch))
    gif.SaveGIF()


if __name__ == "__main__":
    main()
