import argparse

import torch
from torch.utils.data import DataLoader

import utils.loader as l

from learnergy.models.bernoulli import ConvRBM
from opytimizer.utils.history import History

def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Evaluates a ConvRBM-based model using best parameters.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['mnist'])

    parser.add_argument('history', help='History object identifier', type=str)

    parser.add_argument('-visible_shape', help='Shape of input units', type=tuple, default=(28, 28))

    parser.add_argument('-n_channels', help='Number of channels', type=int, default=1)

    parser.add_argument('-n_classes', help='Number of classes', type=int, default=10)

    parser.add_argument('-steps', help='Number of CD steps', type=int, default=1)

    parser.add_argument('-batch_size', help='Batch size', type=int, default=128)

    parser.add_argument('-epochs', help='Number of training epochs', type=int, default=5)

    parser.add_argument('-device', help='CPU or GPU usage', choices=['cpu', 'cuda'])

    parser.add_argument('-seed', help='Seed identifier', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering common variables
    dataset = args.dataset
    history = args.history
    seed = args.seed

    # Gathering RBM-related variable
    visible_shape = args.visible_shape
    n_channels = args.n_channels
    n_classes = args.n_classes
    steps = args.steps
    batch_size = args.batch_size
    epochs = args.epochs
    device = args.device

    # Checks for the name of device
    if device == 'cpu':
        # Updates accordingly
        use_gpu = False
    else:
        # Updates accordingly
        use_gpu = True

    # Loads the data
    train, _, test = l.load_dataset(name=dataset, seed=seed)

    # Defining seeds
    torch.manual_seed(seed)

    # Creates an empty History object and loads the file
    h = History()
    h.load(history)

    # Best parameters
    filter_shape = (int(h.best_agent[-1][0][0][0]), int(h.best_agent[-1][0][0][0]))
    n_filters = int(h.best_agent[-1][0][1][0])
    lr = h.best_agent[-1][0][2][0]
    momentum = h.best_agent[-1][0][3][0]
    decay = h.best_agent[-1][0][4][0]

    # Initializes the model
    model = ConvRBM(visible_shape, filter_shape, n_filters, n_channels, steps, lr, momentum, decay, use_gpu)
    
    # Trains the model using the training set
    model.fit(train, batch_size, epochs)

    # Reconstructs over the testing set
    mse, _ = model.reconstruct(test)

    # Saves the evaluated model
    model.save('crbm.pth')
