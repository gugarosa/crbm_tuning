import argparse

import numpy as np
import torch

import utils.loader as l
import utils.objects as o
import utils.optimizer as opt
import utils.target as t


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Optimizes a ConvRBM-based model using standard meta-heuristics.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['mnist'])

    parser.add_argument('mh', help='Meta-heuristic identifier', choices=['ga'])

    parser.add_argument('-visible_shape', help='Shape of input units', nargs='+', type=int, default=[28, 28])

    parser.add_argument('-n_channels', help='Number of channels', type=int, default=1)

    parser.add_argument('-steps', help='Number of CD steps', type=int, default=1)

    parser.add_argument('-batch_size', help='Batch size', type=int, default=128)

    parser.add_argument('-epochs', help='Number of training epochs', type=int, default=5)

    parser.add_argument('-device', help='CPU or GPU usage', choices=['cpu', 'cuda'])

    parser.add_argument('-n_agents', help='Number of meta-heuristic agents', type=int, default=10)

    parser.add_argument('-n_iter', help='Number of meta-heuristic iterations', type=int, default=15)

    parser.add_argument('-seed', help='Seed identifier', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering common variables
    dataset = args.dataset
    seed = args.seed

    # Gathering RBM-related variable
    visible_shape = tuple(args.visible_shape)
    n_channels = args.n_channels
    steps = args.steps
    batch_size = args.batch_size
    epochs = args.epochs
    device = args.device

    # Gathering optimization variables
    n_agents = args.n_agents
    n_iterations = args.n_iter
    mh_name = args.mh
    mh = o.get_mh(mh_name).obj
    hyperparams = o.get_mh(args.mh).hyperparams

    # Checks for the name of device
    if device == 'cpu':
        # Updates accordingly
        use_gpu = False
    else:
        # Updates accordingly
        use_gpu = True

    # Loads the data
    train, val, _ = l.load_dataset(name=dataset, seed=seed)

    # Defining seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Defines the optimization task
    opt_fn = t.reconstruction(visible_shape, n_channels, steps, use_gpu, batch_size, epochs, train, val)

    # Defines the boundaries
    # [filter_shape, n_filters, lr, momentum, decay]
    lb = [1, 1, 0.0, 0.0, 0.0]
    ub = [7, 10, 1.0, 1.0, 1.0]

    # Running the optimization task
    history = opt.standard_opt(mh, opt_fn, n_agents, len(lb), n_iterations, lb, ub, hyperparams)

    # Saving history object
    history.save(f'{mh_name}.pkl')
