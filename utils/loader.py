import random

import torch
import torchvision as tv
import scipy.io as sio

# A constant used to hold a dictionary of possible datasets
DATASETS = {
    'mnist': tv.datasets.MNIST
    'caltech': 
}


def load_dataset(name='mnist', size=(28, 28), val_split=0.2, seed=0):
    """Loads an input dataset.

    Args:
        name (str): Name of dataset to be loaded.
        size (tuple): Height and width to be resized.
        val_split (float): Percentage of split for the validation set.
        seed (int): Random seed.

    Returns:
        Training, validation and testing sets of loaded dataset.

    """

    # Defining the torch seed
    torch.manual_seed(seed)

    # Loads the training data
    train = DATASETS[name](root='./data', train=True, download=True,
                           transform=tv.transforms.Compose(
                               [tv.transforms.ToTensor(),
                                tv.transforms.Resize(size)])
                           )

    # Splitting the training data into training/validation
    train, val = torch.utils.data.random_split(
        train, [int(len(train) * (1 - val_split)), int(len(train) * val_split)])

    # Loads the testing data
    test = DATASETS[name](root='./data', train=False, download=True,
                          transform=tv.transforms.Compose(
                              [tv.transforms.ToTensor(),
                               tv.transforms.Resize(size)])
                          )

    return train, val, test

def load_caltech101silhouettes():
    caltech_raw = sio.loadmat('./dataset/Caltech101Silhouettes/caltech101_silhouettes_28_split1.mat')

    # train, validation and test data
    x_train = caltech_raw['train_data'].astype('float32').reshape((-1, 28, 28))
    x_val = caltech_raw['val_data'].astype('float32').reshape((-1, 28, 28))
    x_test = caltech_raw['test_data'].astype('float32').reshape((-1, 28, 28))

    y_train = caltech_raw['train_labels']
    y_val = caltech_raw['val_labels']
    y_test = caltech_raw['test_labels']

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))

    return train_loader, val_loader, test_loader