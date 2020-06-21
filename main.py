
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from matplotlib.patches import Ellipse
from scipy.spatial.distance import squareform
from sklearn import datasets, manifold
from sklearn.metrics.pairwise import pairwise_distances

import dataloader
import model.model_tsne_nn as model_tsne_nn
import model.model_tsne as model_tsne

matplotlib.use('Agg')


def train(args, Model, Loss, device, data, target, optimizer, epoch):

    BATCH_SIZE = args.batch_size

    # batch_idx = 0
    Model.train()
    data, target = data.to(device), target.to(device)

    num_train_sample = data.shape[0]
    num_batch = (num_train_sample-0.5)//BATCH_SIZE + 1
    rand_index_i = torch.randperm(num_train_sample)
    rand_index_j = torch.randperm(num_train_sample)

    for batch_idx in torch.arange(0, num_batch):

        start = (batch_idx * BATCH_SIZE).int()
        end = torch.min(
            torch.tensor(
                [batch_idx * BATCH_SIZE + BATCH_SIZE, num_train_sample]
            )
        )
        sample_index_i = rand_index_i[start: end.int()]
        sample_index_j = rand_index_j[start: end.int()]

        optimizer.zero_grad()

        loss = Model(sample_index_i, sample_index_j)
        loss.backward()
        optimizer.step()

    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), 1,
        100. * batch_idx / 1, loss.item()))


def test(args, Model, Loss, device, data, target):
    Model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = Model(data)
            # sum up batch loss
            test_loss += Loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def GetParam():

    parser = argparse.ArgumentParser(description='zelin zang author')
    parser.add_argument('--data_name', type=str,
                        default='digits', metavar='N',)
    parser.add_argument('--data_trai_n', type=int, default=10000, metavar='N',)
    parser.add_argument('--data_test_n', type=int, default=10000, metavar='N',)

    parser.add_argument('--perplexity', type=int, default=30, metavar='N',)

    parser.add_argument('--batch_size', type=int, default=10000, metavar='N',)
    parser.add_argument('--epochs', type=int, default=30000, metavar='N')
    parser.add_argument('--lr', type=float, default=10.00, metavar='LR',)
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',)
    parser.add_argument('--no-cuda', action='store_true', default=False,)
    parser.add_argument('--seed', type=int, default=1, metavar='S',)
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',)
    parser.add_argument('--save-model', action='store_true', default=False,)
    args = parser.parse_args()

    args.batch_size = min(args.batch_size, args.data_trai_n, args.data_test_n,)

    return args


def main():

    args = GetParam()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    data_train, data_test, label_train, label_test = dataloader.GetData(
        args, device=device)

    Model = model_tsne_nn.TSNE_NN(data_train, device=device, args=args).to(device)
    # Model = model_tsne.TSNE(data_train, device=device, args=args).to(device)
    Loss = None  # model.TSNE_LOSS().to(device)
    optimizer = optim.Adadelta(Model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, Model, Loss, device, data_train,
              label_train, optimizer, epoch)
        # test(args, Model, Loss, device, loader_test)
        if epoch % args.log_interval == 0:
            em = Model.GetEmbedding()
            plt.scatter(em[:, 0], em[:, 1], c=label_train.detach().cpu())
            plt.savefig('pic/tsne.png')
            plt.close()


if __name__ == "__main__":
    main()
