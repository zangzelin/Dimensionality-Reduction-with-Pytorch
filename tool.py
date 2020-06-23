import numpy as np
import matplotlib.pyplot as plt
import imageio
import random as rd
import time
import torch
import os


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def LearningRateScheduler(loss_his, optimizer, lr_base):

    loss_his = np.array(loss_his)
    num_shock = np.sum((loss_his[:-1] - loss_his[1:]) < 0)
    if num_shock > 0.40 * loss_his.shape[0]:
        lr_new = lr_base*0.8
        adjust_learning_rate(optimizer, lr_new)
    # elif num_shock < 0.02 * loss_his.shape[0]:
    #     lr_new = lr_base*1.5
    #     adjust_learning_rate(optimizer, lr_new)
    else:
        lr_new = lr_base
    print('*** lr {} -> {}  ***'.format(lr_base, lr_new))
    print('num_shock', num_shock)
    return lr_new


class GIFPloter():
    def __init__(self,):
        self.path_list = []

    def PlotOtherLayer(self, fig,
                       data, label,
                       title='',
                       fig_position0=1,
                       fig_position1=1,
                       fig_position2=1,
                       s=10):
        from sklearn.decomposition import PCA

        color_list = []
        for i in range(label.shape[0]):
            color_list.append(int(label[i]))

        if data.shape[1] > 3:
            pca = PCA(n_components=2)
            data_em = pca.fit_transform(data)
        else:
            data_em = data

        data_em = data_em-data_em.mean(axis=0)

        if data_em.shape[1] == 3:
            ax = fig.add_subplot(fig_position0, fig_position1,
                                 fig_position2, projection='3d')

            ax.scatter(
                data_em[:, 0], data_em[:, 1], data_em[:, 2],
                c=color_list, s=s, cmap='rainbow')

        if data_em.shape[1] == 2:
            ax = fig.add_subplot(fig_position0, fig_position1, fig_position2)
            ax.scatter(
                data_em[:, 0], data_em[:, 1], c=label, s=s, cmap='rainbow')
            plt.axis('equal')

        plt.title(title)

    def AddNewFig(self, latent, label, his_loss=None, title_='', path='./'):

        fig = plt.figure(figsize=(20, 10))

        self.PlotOtherLayer(
            fig, latent,
            label, title=title_,
            fig_position0=1,
            fig_position1=2,
            fig_position2=1)

        fig.add_subplot(1, 2, 2)
        plt.plot(his_loss)
        plt.title('loss history')

        plt.tight_layout()
        path_c = path+title_
        self.path_list.append(path_c)
        plt.savefig(path_c)
        plt.close()

        # data = output_info[9]
        # Analizer.PlotHist(data, title_)

    def SaveGIF(self):

        gif_images = []
        for i, path_ in enumerate(self.path_list):
            # print(path_)
            gif_images.append(imageio.imread(path_))
            if i > 0 and i < len(self.path_list)-2:
                os.remove(path_)

        imageio.mimsave(path_[:-4]+".gif", gif_images, fps=10)


def SetSeed(seed):
    """function used to set a random seed

    Arguments:
        seed {int} -- seed number, will set to torch, random and numpy
    """
    SEED = seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    rd.seed(SEED)
    np.random.seed(SEED)


def GetPath(name=''):

    rest = time.strftime("%Y%m%d%H%M%S_", time.localtime()) + \
        os.popen('git rev-parse HEAD').read()
    path = 'log/' + rest[:20] + name
    if not os.path.exists(path):
        os.makedirs(path)

    return path+'/'


def SaveParam(path, param):
    for v, k in param.__dict__.items():
        print('{v}:{k}'.format(v=v, k=k))
        print('{v}:{k}'.format(v=v, k=k), file=open(path+'/param.txt', 'a'))


def ModelSaver(model, path, name):
    torch.save(model.state_dict(), path+name+'.model')


def ModelLoader(model, path, name):
    model.load_state_dict(torch.load(path+name+'.model'))
