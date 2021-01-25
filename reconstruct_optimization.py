import argparse

import torch
from torch.utils.data import DataLoader

import utils.loader as l
import utils.objects as o


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Optimizes an RBM-based model.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['mnist'])

    parser.add_argument('model_name', help='Model identifier', choices=['drbm', 'rbm'])

    parser.add_argument('-n_input', help='Number of input units', type=int, default=784)

    parser.add_argument('-n_hidden', help='Number of hidden units', type=int, default=128)

    parser.add_argument('-n_classes', help='Number of classes', type=int, default=10)

    parser.add_argument('-steps', help='Number of CD steps', type=int, default=1)

    parser.add_argument('-lr', help='Learning rate', type=float, default=0.1)

    parser.add_argument('-momentum', help='Momentum', type=float, default=0)

    parser.add_argument('-decay', help='Weight decay', type=float, default=0)

    parser.add_argument('-temperature', help='Temperature', type=float, default=1)

    parser.add_argument('-batch_size', help='Batch size', type=int, default=128)

    parser.add_argument('-epochs', help='Number of training epochs', type=int, default=5)

    parser.add_argument('-device', help='CPU or GPU usage', choices=['cpu', 'cuda'])

    parser.add_argument('-seed', help='Seed identifier', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    name = args.model_name
    output = f'{name}.pth'
    n_input = args.n_input
    n_hidden = args.n_hidden
    n_classes = args.n_classes
    steps = args.steps
    lr = args.lr
    momentum = args.momentum
    decay = args.decay
    T = args.temperature
    batch_size = args.batch_size
    epochs = args.epochs
    device = args.device
    seed = args.seed

    # Checks for the name of device
    if device == 'cpu':
        # Updates accordingly
        use_gpu = False
    else:
        # Updates accordingly
        use_gpu = True

    # Loads the data
    train, _, _ = l.load_dataset(name=dataset, seed=seed)

    # Defining the torch seed
    torch.manual_seed(seed)

    # Gathering the model
    model_obj = o.get_model(name).obj

    # Checks the model
    if name == 'rbm':
        # Initializes accordingly
        model = model_obj(n_visible=n_input, n_hidden=n_hidden, steps=steps, learning_rate=lr,
                          momentum=momentum, decay=decay, temperature=T, use_gpu=use_gpu)
    
    elif name == 'drbm':
        # Initializes accordingly
        model = model_obj(n_visible=n_input, n_hidden=n_hidden, n_classes=n_classes, steps=steps,
                          learning_rate=lr, momentum=momentum, decay=decay, temperature=T, use_gpu=use_gpu)

    # Fitting the model
    model.fit(train, batch_size=batch_size, epochs=epochs)

    # Saving model
    torch.save(model, output)

    print(model.history)
