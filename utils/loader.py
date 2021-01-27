import random

import scipy.io as sio
import torch
import torchvision as tv
from torch.utils.data import TensorDataset

# A constant used to hold a dictionary of possible datasets
DATASETS = {
    'fmnist': tv.datasets.FashionMNIST,
    'kmnist': tv.datasets.KMNIST,
    'mnist': tv.datasets.MNIST,
    'semeion': tv.datasets.SEMEION
}


def load_caltech101():
    """Loads the Caltech101 Silhouettes.

    Returns:
        Training, validation and testing sets of Caltech101 Silhouettes.

    """

    # Loads the dataset from a .mat file
    data = sio.loadmat('./datasets/caltech101/silhouettes.mat')

    # Gathers the samples, put them as `float` and reshapes them to 4D
    x_train = data['train_data'].astype('float32').reshape((-1, 1, 28, 28))
    x_val = data['val_data'].astype('float32').reshape((-1, 1, 28, 28))
    x_test = data['test_data'].astype('float32').reshape((-1, 1, 28, 28))

    # Gathers the labels, put them as `long`, squeezes the last dimension
    # and subtract 1 to make their values start from zero
    y_train = data['train_labels'].astype('long').squeeze(-1) - 1
    y_val = data['val_labels'].astype('long').squeeze(-1) - 1
    y_test = data['test_labels'].astype('long').squeeze(-1) - 1

    # Loads the sets using TensorDataset
    train = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    test = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    return train, val, test


def load_semeion(split=0.25):
    """Loads the Semeion dataset.

    Args:
        split (float): Percentage of split for both validation and test set.

    Returns:
        Training, validation and testing sets of Semeion.

    """

    # Loads the data
    data = DATASETS['semeion'](root='./data', download=True,
                               transform=tv.transforms.Compose([tv.transforms.ToTensor()]))

    # Splitting the data into training/validation/test
    train, val, test = torch.utils.data.random_split(data, [int(
        len(data) * (1 - 2 * split) + 1), int(len(data) * split), int(len(data) * split)])

    return train, val, test


def load_dataset(name='mnist', size=(28, 28), val_split=0.25, seed=0):
    """Loads a dataset.

    Args:
        name (str): Name of dataset to be loaded.
        size (tuple): Height and width to be resized.
        val_split (float): Percentage of split for the validation set.
        seed (int): Randomness seed.

    Returns:
        Training, validation and testing sets of loaded dataset.

    """

    # Defining the torch seed
    torch.manual_seed(seed)

    # Checks if it is supposed to load custom datasets
    if name == 'caltech101':
        return load_caltech101()
    elif name == 'semeion':
        return load_semeion(val_split)

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
