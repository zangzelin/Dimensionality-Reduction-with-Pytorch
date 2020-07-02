import torch
# from torchvision import datasets, transforms
from sklearn import manifold, datasets
from sklearn.datasets import fetch_openml
import numpy as np
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def GetData(args, device, batch_size=128, pca=None):
    if args.data_name == 'mnist':
        X, y = fetch_openml('mnist_784', data_home='~/scikit_learn_data',
                            version=1, return_X_y=True)
        X = X/255
        y = y.astype(np.int32)

        index = (
            y < args.numberClass
        )
        X = X[index]
        y = y[index]

        n1 = args.data_trai_n
        n2 = 60000
        data_train, data_test = X[:n1, :], X[n1:n2, :]
        label_train, label_test = y[:n1], y[n1:n2]

        if pca is not None:
            from sklearn.decomposition import PCA
            clf = PCA(n_components=64)
            clf.fit(data_train)
            data_train = clf.transform(data_train)
            data_test = clf.transform(data_test)

    if args.data_name == 'digits':
        digitsr = datasets.load_digits(n_class=6)
        data = digitsr.data/255 * 2 - 1
        label = digitsr.target
        n1 = args.data_trai_n
        n2 = args.data_trai_n + args.data_test_n
        data_train, data_test = data[:n1, :], data[n1:n2, :]
        label_train, label_test = label[:n1], label[n1:n2]

    data_train = torch.tensor(data_train, device=device)
    data_test = torch.tensor(data_test, device=device)
    label_train = torch.tensor(label_train, device=device)
    label_test = torch.tensor(label_test, device=device)
    return data_train, data_test, label_train, label_test


def GetDataTwo(args, device, a, b, batch_size=128, pca=None):

    if args.data_name == 'mnist':
        X, y = fetch_openml('mnist_784', data_home='~/scikit_learn_data',
                            version=1, return_X_y=True)
        X = X/255
        y = y.astype(np.int32)

        index = ((y == a)+(y == b) > 0)
        X = X[index]
        y = y[index]

        n1 = args.data_trai_n
        n2 = 60000
        data_train, data_test = X[:n1, :], X[n1:n2, :]
        label_train, label_test = y[:n1], y[n1:n2]

        if pca is not None:
            from sklearn.decomposition import PCA
            clf = PCA(n_components=64)
            clf.fit(data_train)
            data_train = clf.transform(data_train)
            data_test = clf.transform(data_test)

    if args.data_name == 'digits':
        digitsr = datasets.load_digits(n_class=6)
        data = digitsr.data/255 * 2 - 1
        label = digitsr.target
        n1 = args.data_trai_n
        n2 = args.data_trai_n + args.data_test_n
        data_train, data_test = data[:n1, :], data[n1:n2, :]
        label_train, label_test = label[:n1], label[n1:n2]

    data_train = torch.tensor(data_train, device=device)
    data_test = torch.tensor(data_test, device=device)
    label_train = torch.tensor(label_train, device=device)
    label_test = torch.tensor(label_test, device=device)
    return data_train, data_test, label_train, label_test

# class FaceLandmarksDataset(Dataset):
#     """Face Landmarks dataset."""

#     def __init__(self, csv_file, root_dir, transform=None):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.landmarks_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.landmarks_frame)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         img_name = os.path.join(self.root_dir,
#                                 self.landmarks_frame.iloc[idx, 0])
#         image = io.imread(img_name)
#         landmarks = self.landmarks_frame.iloc[idx, 1:]
#         landmarks = np.array([landmarks])
#         landmarks = landmarks.astype('float').reshape(-1, 2)
#         sample = {'image': image, 'landmarks': landmarks}

#         if self.transform:
#             sample = self.transform(sample)

#         return sample
