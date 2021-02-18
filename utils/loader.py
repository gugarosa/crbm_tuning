import os
import random
import urllib.request

import numpy as np
import scipy.io as sio
import torch
import torchvision as tv
from torch.utils.data import Dataset, TensorDataset

# A constant used to hold a dictionary of possible datasets
DATASETS = {
    'fmnist': tv.datasets.FashionMNIST,
    'kmnist': tv.datasets.KMNIST,
    'mnist': tv.datasets.MNIST,
    'semeion': tv.datasets.SEMEION
}


class Mpeg7Dataset(Dataset):
    """Loads the Natural images from MPEG-7 Core Experiment.

    """

    def __init__(self, root, split_data='', transform=None):
        """Initialization method.

        Args:
            root (string): Path to the .mat file.
            split_data (string): Optional train, val, test.
            transform (callable): Optional transform to be applied on a sample.

        """

        # Loads the dataset from a .mat file
        data = sio.loadmat(root)         
        
        # Gathers the samples
        self.data = data['data_' + split_data].squeeze()

        # Gathers the labels 
        self.labels = data['labels_' + split_data].squeeze()

        # Defines the transform
        self.transform = transform

    def __len__(self):
        """Returns the length of the dataset.

        """
        
        return len(self.data)

    def __getitem__(self, idx):
        """Returns each individual sample of the dataset.

        Args:
            idx (int): Sample identifier.

        """

        # Checks if index is a tensor
        if torch.is_tensor(idx):
            # If yes, transforms to a list
            idx = idx.tolist()

        # Gathers the sample
        sample = self.data[idx]

        # If there is a pre-defined transform
        if self.transform:
            # Applies the transform
            sample = self.transform(sample)

        return (sample, self.labels[idx])

class NaturalImagesDataset(Dataset):
    """Loads the Natural images from Olshausen's.

    """

    def __init__(self, root, transform=None):
        """Initialization method.

        Args:
            root (string): Path to the .mat file.
            transform (callable): Optional transform to be applied on a sample.

        """

        # Loads the dataset from a .mat file
        data = sio.loadmat(root)

        # Gathers the samples, put them as `float` and reshape them
        self.data = data['IMAGES'].astype('float32').reshape((512, 512, 1, -1))

        # Creates labels from each image
        self.labels = np.arange(len(self.data))

        # Defines the transform
        self.transform = transform

    def __len__(self):
        """Returns the length of the dataset.

        """

        return self.data.shape[3]

    def __getitem__(self, idx):
        """Returns each individual sample of the dataset.

        Args:
            idx (int): Sample identifier.

        """

        # Checks if index is a tensor
        if torch.is_tensor(idx):
            # If yes, transforms to a list
            idx = idx.tolist()

        # Gathers the sample
        sample = self.data[:, :, :, idx]

        # If there is a pre-defined transform
        if self.transform:
            # Applies the transform
            sample = self.transform(sample)

        return (sample, self.labels[idx])


def download_file(url, output_path):
    """Downloads a file given its URL and the output path to be saved.

    Args:
        url (str): URL to download the file.
        output_path (str): Path to save the downloaded file.

    """

    # Checks if file exists
    file_exists = os.path.exists(output_path)

    # If file does not exist
    if not file_exists:
        # Downloads the file
        urllib.request.urlretrieve(url, output_path)


def load_caltech101(size):
    """Loads the Caltech101 Silhouettes.

    Args:
        size (tuple): Height and width to be resized.

    Returns:
        Training, validation and testing sets of Caltech101 Silhouettes.

    """

    # Attempts to download the file
    output_path = './data/caltech101_silhouettes.mat'
    download_file('http://recogna.tech/files/crbm_tuning/caltech101_silhouettes.mat', output_path)

    # Loads the dataset from a .mat file
    data = sio.loadmat(output_path)

    # Gathers the samples, put them as `float` and reshapes them to 4D
    x_train = data['train_data'].astype('float32').reshape((-1, 1, 28, 28))
    x_val = data['val_data'].astype('float32').reshape((-1, 1, 28, 28))
    x_test = data['test_data'].astype('float32').reshape((-1, 1, 28, 28))

    # Resizing data
    x_train = np.resize(x_train, ((x_train.shape[0], 1, *size)))
    x_val = np.resize(x_val, ((x_val.shape[0], 1, *size)))
    x_test = np.resize(x_test, ((x_test.shape[0], 1, *size)))

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

def load_mpeg7(size):
    """Loads the MPEG-7 Core Experiment.

    Args:
        size (tuple): Height and width to be resized.

    Returns:
        Training, validation and testing sets of MPEG-7 Core Experiment.

    """

    # Attempts to download the file
    output_path = './data/mpeg7.mat'
    download_file('http://recogna.tech/files/crbm_tuning/mpeg7.mat', output_path)

    # Defining a dictionary of transforms
    data_transforms = {
            'train': tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.RandomResizedCrop(size),
                tv.transforms.RandomHorizontalFlip()
            ]),
            'val': tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Resize(size)
            ]),
    }
    
    # Loads the sets using MPEG7Dataset   
    train = Mpeg7Dataset(root=output_path, split_data='train', transform=data_transforms['train'])
    val = Mpeg7Dataset(root=output_path, split_data='val', transform=data_transforms['val'])
    test = Mpeg7Dataset(root=output_path, split_data='test', transform=data_transforms['val'])

    return train, val, test


def load_natural_images(size):
    """Loads the Natural Images dataset.

    Args:
        size (tuple): Height and width to be resized.

    Returns:
        Training, validation and testing sets of Natural Images.

    """

    # Attempts to download the file
    output_path = './data/natural_images.mat'
    download_file('http://recogna.tech/files/crbm_tuning/natural_images.mat', output_path)

    # Defining a dictionary of transforms
    data_transforms = {
        'train': tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.RandomResizedCrop(size),
            tv.transforms.RandomHorizontalFlip()
        ]),
        'val': tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Resize(size)
        ]),
    }

    # Loads the sets using NaturalImagesDataset
    train = NaturalImagesDataset(root=output_path, transform=data_transforms['train'])
    val = NaturalImagesDataset(root=output_path, transform=data_transforms['val'])
    test = NaturalImagesDataset(root=output_path, transform=data_transforms['val'])

    return train, val, test


def load_semeion(size, split):
    """Loads the Semeion dataset.

    Args:
        size (tuple): Height and width to be resized.
        split (float): Percentage of split for both validation and test set.

    Returns:
        Training, validation and testing sets of Semeion.

    """

    # Loads the data
    data = DATASETS['semeion'](root='./data', download=True,
                               transform=tv.transforms.Compose(
                                   [tv.transforms.ToTensor(),
                                    tv.transforms.Resize(size)])
                               )

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
        return load_caltech101(size)
    elif name == 'natural_images':
        return load_natural_images(size)
    elif name == 'semeion':
        return load_semeion(size, val_split)
    elif name == 'mpeg7':
        return load_mpeg7(size)

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
