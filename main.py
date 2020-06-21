
import numpy as np
import matplotlib.pyplot as plt
import dataloader
import torch
import torch.optim as optim
import argparse
import model.model_tsne as model

from sklearn import manifold, datasets
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform
from matplotlib.patches import Ellipse

import matplotlib
matplotlib.use('Agg')


def train(args, Model, Loss, device, data, target, optimizer, epoch):
    
    # batch_idx = 0
    Model.train()
    data, target = data.to(device), target.to(device)

    num_train_sample = train_data.shape[0]
    num_batch = (num_train_sample-0.5)//batch_size + 1

    for batch_n in range(num_batch):

        batch_index_list
        data_batch = data[batch_index_list]
        label_batch = target[batch_index_list]

        optimizer.zero_grad()

        loss = Model()
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
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=30000, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=500.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    return args


def main():

    args = GetParam()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    data_train, data_test, label_train, label_test = dataloader.GetData(
        data_name='mnist', device = device)

    Model = model.TSNE(data_train, device=device).to(device)
    Loss = None  # model.TSNE_LOSS().to(device)
    optimizer = optim.Adadelta(Model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, Model, Loss, device, data_train, label_train, optimizer, epoch)
        # test(args, Model, Loss, device, loader_test)
        if epoch % 1000 == 0:
            em = Model.GetEmbedding()
            plt.scatter(em[:, 0], em[:, 1], c=label_train.detach().cpu())
            plt.savefig('lll.png')
            plt.close()


if __name__ == "__main__":
    main()
