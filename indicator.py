import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score


def Loadcsv(path):
    return pd.read_csv(path, delimiter=',').to_numpy()[:, 1:]


def PlotOtherLayer(fig,
                   data, label,
                   title='',
                   fig_position0=1,
                   fig_position1=1,
                   fig_position2=1,
                   s=4):

    from sklearn.decomposition import PCA

    # color_list = []
    # for i in range(label.shape[0]):
    #     color_list.append(int(label[i]))

    if data.shape[1] > 3:
        pca = PCA(n_components=2)
        data_em = pca.fit_transform(data)
    else:
        data_em = data

    data_em = data_em-data_em.mean(axis=0)

    if data_em.shape[1] == 3:
        ax = fig.add_subplot(
            fig_position0, fig_position1,
            fig_position2, projection='3d')

        ax.scatter(
            data_em[:, 0], data_em[:, 1], data_em[:, 2],
            c=label, s=s, cmap='rainbow')

    if data_em.shape[1] == 2:
        ax = fig.add_subplot(fig_position0, fig_position1, fig_position2)
        ax.scatter(
            data_em[:, 0], data_em[:, 1], c=label, s=s, cmap='rainbow')
        if title in ['HLLE', 'LTSA', 'MLLE']:
            ax.scatter(
                [0, 0], [-0.075*4, 0.075*4], c='w', s=0.1)
        else:
            ax.axis('equal')
            print('--')

            # plt.ylim([-0.075*4, 0.075*4])

    plt.title(title)


def Srotate_onepoint(angle, valuex, valuey, pointx, pointy):
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    sRotatex = (valuex-pointx)*math.cos(angle) + \
        (valuey-pointy)*math.sin(angle) + pointx
    sRotatey = (valuey-pointy)*math.cos(angle) - \
        (valuex-pointx)*math.sin(angle) + pointy
    return sRotatex, sRotatey


def Srotate(angle, data):
    for i in range(data.shape[0]):
        data[i, 0], data[i, 1] = Srotate_onepoint(
            angle, data[i, 0], data[i, 1], 0, 0)

    return data


def get_w(data):
    num = data.shape[0]
    A = np.ones((num, 3))
    b = np.zeros((num, 1))

    A[:, 0:2] = data[:, 0:2]
    b[:, 0:1] = data[:, 2:3]

    A_T = A.T
    A1 = np.dot(A_T, A)

    if np.linalg.matrix_rank(A1) == 3:
        A2 = np.linalg.inv(A1)
        A3 = np.dot(A2, A_T)
        X = np.dot(A3, b)

        w = np.zeros(4)
        w[0] = X[0, 0]
        w[1] = X[1, 0]
        w[2] = -1
        w[3] = X[2, 0]

    else:
        w = None

    return w


def project(data, w):
    if w is None:
        return 0
    else:
        A = w[0]
        B = w[1]
        C = w[2]
        D = w[3]
        dis = 0

        for i in range(data.shape[0]):
            p = data[i]
            out = np.zeros(3)

            out[0] = ((B**2 + C**2)*p[0] - A*(B*p[1] + C*p[2] + D)) / \
                (A**2 + B**2 + C**2)
            out[1] = ((A**2 + C**2)*p[1] - B*(A*p[0] + C*p[2] + D)) / \
                (A**2 + B**2 + C**2)
            out[2] = ((A**2 + B**2)*p[2] - C*(A*p[0] + B*p[1] + D)) / \
                (A**2 + B**2 + C**2)

            dis += np.linalg.norm(out - p, ord=2)

        return dis / data.shape[0]


def pro_error_calc(lat):
    scale = np.max(pdist(lat))
    lat = lat/scale
    w = get_w(lat)
    dis = project(lat, w)

    return dis


def CalPairwiseDis(data, neighbors):

    dis_list = []
    for i in range(data.shape[0]):
        for j in range(neighbors.shape[1]):
            m = int(neighbors[i, j])
            dis = np.linalg.norm(
                data[i] - data[m], ord=2) / (data.shape[1] ** 0.5)
            dis_list.append(dis)

    dis_list = np.array(dis_list)
    return dis_list


def Neighbor(data, k=5):
    num = data.shape[0]
    dists = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            dists[i, j] = np.linalg.norm(data[i] - data[j], ord=2)

    neighbors = np.zeros((num, k))

    for i in range(num):
        count = 0
        index = np.argsort(dists[i, :])
        for j in range(num):
            if count < k:
                if i != index[j]:
                    neighbors[i, count] = index[j]
                    count += 1
            else:
                break

    return neighbors


def Lipschitz(x1, x2, K=5):
    neighbors = Neighbor(x1, k=K)
    dis_list_old = CalPairwiseDis(x1, neighbors)
    dis_list = CalPairwiseDis(x2, neighbors)
    dis = dis_list / dis_list_old

    dis_list = []
    for j in range(len(dis)//K):
        dis_list.append(
            max(np.max(dis[j*K:j*K+K]), 1.0/np.min(dis[j*K:j*K+K])))

    dis_list = np.array(dis_list)

    return np.min(dis_list), np.max(dis_list)


def GetIndicator(real_data, latent, lat=None, rcons=None,  KNN=5, k=10):

    if torch.is_tensor(real_data):
        real_data = real_data.detach().cpu().numpy()
    if torch.is_tensor(latent):
        latent = latent.detach().cpu().numpy()
    if lat is not None:
        # if torch.is_tensor(lat):
        #     lat = lat.detach().cpu().numpy()
        if len(lat) == 1:
            pro_error = pro_error_calc(lat[0])
        else:
            pro_error1 = pro_error_calc(lat[0])
            pro_error2 = pro_error_calc(lat[1])
            pro_error = (pro_error2 + pro_error1)/2
            print(pro_error1, pro_error2, pro_error)

    # real_data = real_data-np.min(real_data)
    # latent = latent-np.min(latent)
    # real_data = real_data/np.max(real_data)
    # latent = latent/np.max(latent)

    calc = MeasureCalculator(real_data, latent, 31)
    print('--')
    rmse = calc.rmse()
    kl1 = calc.density_kl_global_1()
    kl01 = calc.density_kl_global_01()
    kl001 = calc.density_kl_global_001()
    print('--')

    rmse_local = []
    mrreZX = []
    mrreXZ = []
    cont = []
    trust = []
    for k in range(4, 10, 1):
        rmse_local.append(calc.local_rmse(k=k))

    for k in range(10, 30, 10):
        mrreZX.append(calc.mrre(k)[0])
        mrreXZ.append(calc.mrre(k)[1])
        cont.append(calc.continuity(k))
        trust.append(calc.trustworthiness(k))

    # Lipschitz_min, Lipschitz_max = Lipschitz(real_data, latent, K=KNN)

    indicator = {}
    indicator['kl001'] = kl001
    indicator['kl01'] = kl01
    indicator['kl1'] = kl1
    indicator['mrre ZX'] = np.mean(mrreZX)
    indicator['mrre XZ'] = np.mean(mrreXZ)
    indicator['cont'] = np.mean(cont)
    indicator['trust'] = np.mean(trust)
    indicator['rmse'] = rmse
    indicator['local rmse'] = np.mean(rmse_local)
    # indicator['L_min'] = Lipschitz_min
    # indicator['L_max'] = Lipschitz_max
    if rcons:
        indicator['chongjian'] = np.sqrt(mean_squared_error(real_data, rcons))

    if lat is not None:
        indicator['projection error'] = pro_error

    # print(indicator)

    return indicator


class MeasureRegistrator():
    """Keeps track of measurements in Measure Calculator."""
    k_independent_measures = {}
    k_dependent_measures = {}

    def register(self, is_k_dependent):
        def k_dep_fn(measure):
            self.k_dependent_measures[measure.__name__] = measure
            return measure

        def k_indep_fn(measure):
            self.k_independent_measures[measure.__name__] = measure
            return measure

        if is_k_dependent:
            return k_dep_fn
        return k_indep_fn

    def get_k_independent_measures(self):
        return self.k_independent_measures

    def get_k_dependent_measures(self):
        return self.k_dependent_measures


class MeasureCalculator():
    measures = MeasureRegistrator()

    def __init__(self, Xi, Zi, k_max):
        self.k_max = k_max
        if torch.is_tensor(Xi):
            self.X = Xi.detach().cpu().numpy()
            self.Z = Zi.detach().cpu().numpy()
        else:
            self.X = Xi
            self.Z = Zi
        self.pairwise_X = squareform(pdist(self.X))
        self.pairwise_Z = squareform(pdist(self.Z))
        self.neighbours_X, self.ranks_X = \
            self._neighbours_and_ranks(self.pairwise_X, k_max)
        self.neighbours_Z, self.ranks_Z = \
            self._neighbours_and_ranks(self.pairwise_Z, k_max)

        print('finish init')

    @staticmethod
    def _neighbours_and_ranks(distances, k):
        """
        Inputs: 
        - distances,        distance matrix [n times n], 
        - k,                number of nearest neighbours to consider
        Returns:
        - neighbourhood,    contains the sample indices (from 0 to n-1) of kth nearest neighbor of current sample [n times k]
        - ranks,            contains the rank of each sample to each sample [n times n], whereas entry (i,j) gives the rank that sample j has to i (the how many 'closest' neighbour j is to i) 
        """
        # Warning: this is only the ordering of neighbours that we need to
        # extract neighbourhoods below. The ranking comes later!
        indices = np.argsort(distances, axis=-1, kind='stable')

        # Extract neighbourhoods.
        neighbourhood = indices[:, 1:k+1]

        # Convert this into ranks (finally)
        ranks = indices.argsort(axis=-1, kind='stable')
        # print(ranks)

        return neighbourhood, ranks

    def get_X_neighbours_and_ranks(self, k):
        return self.neighbours_X[:, :k], self.ranks_X

    def get_Z_neighbours_and_ranks(self, k):
        return self.neighbours_Z[:, :k], self.ranks_Z

    def compute_k_independent_measures(self):
        return {key: fn(self) for key, fn in
                self.measures.get_k_independent_measures().items()}

    def compute_k_dependent_measures(self, k):
        return {key: fn(self, k) for key, fn in
                self.measures.get_k_dependent_measures().items()}

    def compute_measures_for_ks(self, ks):
        return {
            key: np.array([fn(self, k) for k in ks])
            for key, fn in self.measures.get_k_dependent_measures().items()
        }

    @measures.register(False)
    def stress(self):
        sum_of_squared_differences = \
            np.square(self.pairwise_X - self.pairwise_Z).sum()
        sum_of_squares = np.square(self.pairwise_Z).sum()

        return np.sqrt(sum_of_squared_differences / sum_of_squares)

    @measures.register(False)
    def rmse(self):
        n = self.pairwise_X.shape[0]
        sum_of_squared_differences = np.square(
            self.pairwise_X - self.pairwise_Z).sum()
        return np.sqrt(sum_of_squared_differences / n**2)

    @measures.register(False)
    def local_rmse(self, k):
        X_neighbors, _ = self.get_X_neighbours_and_ranks(k)
        mses = []
        n = self.pairwise_X.shape[0]
        for i in range(n):
            x = self.X[X_neighbors[i]]
            z = self.Z[X_neighbors[i]]
            d1 = np.sqrt(
                np.square(x - self.X[i]).sum(axis=1))/np.sqrt(self.X.shape[1])
            d2 = np.sqrt(
                np.square(z - self.Z[i]).sum(axis=1))/np.sqrt(self.Z.shape[1])
            mse = np.sum(np.square(d1 - d2))
            mses.append(mse)
        return np.sqrt(np.sum(mses)/(k*n))

    @staticmethod
    def _trustworthiness(X_neighbourhood, X_ranks, Z_neighbourhood,
                         Z_ranks, n, k):
        '''
        Calculates the trustworthiness measure between the data space `X`
        and the latent space `Z`, given a neighbourhood parameter `k` for
        defining the extent of neighbourhoods.
        '''

        result = 0.0

        # Calculate number of neighbours that are in the $k$-neighbourhood
        # of the latent space but not in the $k$-neighbourhood of the data
        # space.
        for row in range(X_ranks.shape[0]):
            missing_neighbours = np.setdiff1d(
                Z_neighbourhood[row],
                X_neighbourhood[row]
            )

            for neighbour in missing_neighbours:
                result += (X_ranks[row, neighbour] - k)

        return 1 - 2 / (n * k * (2 * n - 3 * k - 1)) * result

    @measures.register(True)
    def trustworthiness(self, k):
        X_neighbourhood, X_ranks = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, Z_ranks = self.get_Z_neighbours_and_ranks(k)
        n = self.pairwise_X.shape[0]
        return self._trustworthiness(X_neighbourhood, X_ranks, Z_neighbourhood,
                                     Z_ranks, n, k)

    @measures.register(True)
    def continuity(self, k):
        '''
        Calculates the continuity measure between the data space `X` and the
        latent space `Z`, given a neighbourhood parameter `k` for setting up
        the extent of neighbourhoods.

        This is just the 'flipped' variant of the 'trustworthiness' measure.
        '''

        X_neighbourhood, X_ranks = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, Z_ranks = self.get_Z_neighbours_and_ranks(k)
        n = self.pairwise_X.shape[0]
        # Notice that the parameters have to be flipped here.
        return self._trustworthiness(Z_neighbourhood, Z_ranks, X_neighbourhood,
                                     X_ranks, n, k)

    @measures.register(True)
    def neighbourhood_loss(self, k):
        '''
        Calculates the neighbourhood loss quality measure between the data
        space `X` and the latent space `Z` for some neighbourhood size $k$
        that has to be pre-defined.
        '''

        X_neighbourhood, _ = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, _ = self.get_Z_neighbours_and_ranks(k)

        result = 0.0
        n = self.pairwise_X.shape[0]

        for row in range(n):
            shared_neighbours = np.intersect1d(
                X_neighbourhood[row],
                Z_neighbourhood[row],
                assume_unique=True
            )

            result += len(shared_neighbours) / k

        return 1.0 - result / n

    @measures.register(True)
    def rank_correlation(self, k):
        '''
        Calculates the spearman rank correlation of the data
        space `X` with respect to the latent space `Z`, subject to its $k$
        nearest neighbours.
        '''

        X_neighbourhood, X_ranks = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, Z_ranks = self.get_Z_neighbours_and_ranks(k)

        n = self.pairwise_X.shape[0]
        # we gather
        gathered_ranks_x = []
        gathered_ranks_z = []
        for row in range(n):
            # we go from X to Z here:
            for neighbour in X_neighbourhood[row]:
                rx = X_ranks[row, neighbour]
                rz = Z_ranks[row, neighbour]
                gathered_ranks_x.append(rx)
                gathered_ranks_z.append(rz)
        rs_x = np.array(gathered_ranks_x)
        rs_z = np.array(gathered_ranks_z)
        coeff, _ = spearmanr(rs_x, rs_z)

        # use only off-diagonal (non-trivial) ranks:
        #inds = ~np.eye(X_ranks.shape[0],dtype=bool)
        #coeff, pval = spearmanr(X_ranks[inds], Z_ranks[inds])
        return coeff

    @measures.register(True)
    def mrre(self, k):
        '''
        Calculates the mean relative rank error quality metric of the data
        space `X` with respect to the latent space `Z`, subject to its $k$
        nearest neighbours.
        '''

        X_neighbourhood, X_ranks = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, Z_ranks = self.get_Z_neighbours_and_ranks(k)

        n = self.pairwise_X.shape[0]

        # First component goes from the latent space to the data space, i.e.
        # the relative quality of neighbours in `Z`.

        mrre_ZX = 0.0
        for row in range(n):
            for neighbour in Z_neighbourhood[row]:
                rx = X_ranks[row, neighbour]
                rz = Z_ranks[row, neighbour]

                mrre_ZX += abs(rx - rz) / rz

        # Second component goes from the data space to the latent space,
        # i.e. the relative quality of neighbours in `X`.

        mrre_XZ = 0.0
        for row in range(n):
            # Note that this uses a different neighbourhood definition!
            for neighbour in X_neighbourhood[row]:
                rx = X_ranks[row, neighbour]
                rz = Z_ranks[row, neighbour]

                # Note that this uses a different normalisation factor
                mrre_XZ += abs(rx - rz) / rx

        # Normalisation constant
        C = n * sum([abs(2*j - n - 1) / j for j in range(1, k+1)])
        return mrre_ZX / C, mrre_XZ / C

    @measures.register(False)
    def density_global(self, sigma=0.1):
        X = self.pairwise_X
        X = X / X.max()
        Z = self.pairwise_Z
        Z = Z / Z.max()

        density_x = np.sum(np.exp(-(X ** 2) / sigma), axis=-1)
        density_x /= density_x.sum(axis=-1)

        density_z = np.sum(np.exp(-(Z ** 2) / sigma), axis=-1)
        density_z /= density_z.sum(axis=-1)

        return np.abs(density_x - density_z).sum()

    @measures.register(False)
    def density_kl_global(self, sigma=0.1):
        X = self.pairwise_X
        X = X / X.max()
        Z = self.pairwise_Z
        Z = Z / Z.max()

        density_x = np.sum(np.exp(-(X ** 2) / sigma), axis=-1)
        density_x /= density_x.sum(axis=-1)

        density_z = np.sum(np.exp(-(Z ** 2) / sigma), axis=-1)
        density_z /= density_z.sum(axis=-1)

        return (density_x * (np.log(density_x) - np.log(density_z))).sum()

    @measures.register(False)
    def density_kl_global_10(self):
        return self.density_kl_global(10.)

    @measures.register(False)
    def density_kl_global_1(self):
        return self.density_kl_global(1.)

    @measures.register(False)
    def density_kl_global_01(self):
        return self.density_kl_global(0.1)

    @measures.register(False)
    def density_kl_global_001(self):
        return self.density_kl_global(0.01)

    @measures.register(False)
    def density_kl_global_0001(self):
        return self.density_kl_global(0.001)


if __name__ == "__main__":
    import numpy as np
    latent_data = np.loadtxt('umap_input.txt')
    input_data = np.loadtxt('umap_latent.txt')
    a = GetIndicator(input_data, latent_data)
    print(a)
