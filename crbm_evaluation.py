import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from learnergy.models.bernoulli import ConvRBM
from opytimizer.utils.history import History
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils.loader as l


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Evaluates a ConvRBM-based model using best parameters.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['caltech101', 'fmnist', 'kmnist',
                                                                       'mnist', 'natural_images', 'semeion'])

    parser.add_argument('history', help='History object identifier', type=str)

    parser.add_argument('-visible_shape', help='Shape of input units', nargs='+', type=int, default=[28, 28])

    parser.add_argument('-n_channels', help='Number of channels', type=int, default=1)

    parser.add_argument('-n_classes', help='Number of classes', type=int, default=10)

    parser.add_argument('-steps', help='Number of CD steps', type=int, default=1)

    parser.add_argument('-batch_size', help='Batch size', type=int, default=128)

    parser.add_argument('-epochs', help='Number of training epochs', type=int, default=5)

    parser.add_argument('-fine_tune_epochs', help='Number of fine-tuning epochs', type=int, default=10)

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
    visible_shape = tuple(args.visible_shape)
    n_channels = args.n_channels
    n_classes = args.n_classes
    steps = args.steps
    batch_size = args.batch_size
    epochs = args.epochs
    fine_tune_epochs = args.fine_tune_epochs
    device = args.device

    # Checks for the name of device
    if device == 'cpu':
        # Updates accordingly
        use_gpu = False
    else:
        # Updates accordingly
        use_gpu = True

    # Loads the data
    train, _, test = l.load_dataset(name=dataset, size=visible_shape, seed=seed)

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
    torch.save(model, 'crbm.pth')

    # Creating the Fully Connected layer to append on top of RBM
    fc = nn.Linear(model.hidden_shape[0] * model.hidden_shape[1] * n_filters, n_classes)

    # Check if model uses GPU
    if model.device == 'cuda':
        # If yes, put fully-connected on GPU
        fc = fc.cuda()

    # Cross-Entropy loss is used for the discriminative fine-tuning
    criterion = nn.CrossEntropyLoss()

    # Creating the optimzers
    optimizer = [optim.Adam(model.parameters(), lr=0.0001),
                 optim.Adam(fc.parameters(), lr=0.001)]

    # Creating training and testing batches
    train_batch = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=0)
    test_batch = DataLoader(test, batch_size=10000, shuffle=False, num_workers=0)

    # For amount of fine-tuning epochs
    for e in range(fine_tune_epochs):
        print(f'Epoch {e+1}/{fine_tune_epochs}')

        # Resetting metrics
        train_loss, test_acc = 0, 0

        # For every possible batch
        for x_batch, y_batch in tqdm(train_batch):
            # For every possible optimizer
            for opt in optimizer:
                # Resets the optimizer
                opt.zero_grad()

            # Checking whether GPU is avaliable and if it should be used
            if model.device == 'cuda':
                # Applies the GPU usage to the data and labels
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

            # Passing the batch down the model
            y = model(x_batch)

            # Reshaping the outputs
            y = y.reshape(x_batch.size(0), model.hidden_shape[0] * model.hidden_shape[1] * n_filters)

            # Calculating the fully-connected outputs
            y = fc(y)
            
            # Calculating loss
            loss = criterion(y, y_batch)
            
            # Propagating the loss to calculate the gradients
            loss.backward()
            
            # For every possible optimizer
            for opt in optimizer:
                # Performs the gradient update
                opt.step()

            # Adding current batch loss
            train_loss += loss.item()
            
        # Calculate the test accuracy for the model:
        for x_batch, y_batch in tqdm(test_batch):
            # Checking whether GPU is avaliable and if it should be used
            if model.device == 'cuda':
                # Applies the GPU usage to the data and labels
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

            # Passing the batch down the model
            y = model(x_batch)

            # Reshaping the outputs
            y = y.reshape(x_batch.size(0), model.hidden_shape[0] * model.hidden_shape[1] * n_filters)

            # Calculating the fully-connected outputs
            y = fc(y)

            # Calculating predictions
            _, preds = torch.max(y, 1)

            # Calculating testing set accuracy
            test_acc = torch.mean((torch.sum(preds == y_batch).float()) / x_batch.size(0))

        print(f'Loss: {train_loss / len(train_batch)} | Test Accuracy: {test_acc}')

    # Saving the fine-tuned model
    torch.save(model, 'crbm_fine_tuned.pth')

    # Checking the model's history
    print(model.history)
