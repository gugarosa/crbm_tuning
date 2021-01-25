import argparse

import torch
from torch.utils.data import DataLoader

import utils.loader as l
import utils.objects as o
import utils.optimizer as opt
import utils.targets as t


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Optimizes an RBM-based model.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['mnist'])

    parser.add_argument('model_name', help='Model identifier', choices=['crbm', 'rbm'])

    parser.add_argument('mh', help='Meta-heuristic identifier', choices=['ga'])

    parser.add_argument('-n_input', help='Number of input units', type=int, default=784)
    
    parser.add_argument('-visible_shape', help='Shape of input units', type=tuple, default=(28, 28))

    parser.add_argument('-n_hidden', help='Number of hidden units', type=int, default=128)

    parser.add_argument('-filter_shape', help='Shape of filter units', type=tuple, default=(7, 7))

    parser.add_argument('-n_filters', help='Number of filters', type=int, default=10)

    parser.add_argument('-n_channels', help='Number of channels', type=int, default=1)

    parser.add_argument('-steps', help='Number of CD steps', type=int, default=1)

    parser.add_argument('-lr', help='Learning rate', type=float, default=0.1)

    parser.add_argument('-momentum', help='Momentum', type=float, default=0)

    parser.add_argument('-decay', help='Weight decay', type=float, default=0)

    parser.add_argument('-temperature', help='Temperature', type=float, default=1)

    parser.add_argument('-batch_size', help='Batch size', type=int, default=128)

    parser.add_argument('-epochs', help='Number of training epochs', type=int, default=5)

    parser.add_argument('-device', help='CPU or GPU usage', choices=['cpu', 'cuda'])

    parser.add_argument('-n_agents', help='Number of meta-heuristic agents', type=int, default=10)

    parser.add_argument('-n_variables', help='Number of variables', type=int, default=3)

    parser.add_argument('-n_iter', help='Number of meta-heuristic iterations', type=int, default=15)

    parser.add_argument('-seed', help='Seed identifier', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering common variables
    dataset = args.dataset
    seed = args.seed

    # Gathering RBM-related variables
    name = args.model_name
    n_input = args.n_input
    visible_shape = args.visible_shape
    n_hidden = args.n_hidden
    filter_shape = args.filter_shape
    n_filters = args.n_filters
    n_channels = args.n_channels
    steps = args.steps
    lr = args.lr
    momentum = args.momentum
    decay = args.decay
    T = args.temperature
    batch_size = args.batch_size
    epochs = args.epochs
    device = args.device

    # Gathering optimization variables
    n_agents = args.n_agents
    n_variables = args.n_variables
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

    # Defining the torch seed
    torch.manual_seed(seed)

    # Defines the optimization task
    opt_fn = t.reconstruction(name, n_input, visible_shape, n_hidden, filter_shape, n_filters, n_channels,
                              steps, lr, momentum, decay, T, use_gpu, batch_size, epochs, train, val)

    # Defines the boundaries
    lb = [0] * n_variables
    ub = [1] * n_variables

    # Running the optimization task
    # if mh_name == 'gp':
        # history = opt.tree_opt(mh, opt_fn, n_agents, n_variables, n_iterations, lb, ub, hyperparams)
    # else:
    history = opt.standard_opt(mh, opt_fn, n_agents, n_variables, n_iterations, lb, ub, hyperparams)

    # Saving history object
    history.save(f'{mh_name}.history')
