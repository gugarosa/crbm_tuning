from learnergy.models.bernoulli import ConvRBM
import utils.objects as o


def reconstruction(visible_shape, n_channels, steps, use_gpu, batch_size, epochs, train, val):
    """Wraps the reconstruction task for optimization purposes.

    Args:
        visible_shape (tuple): Shape of visible units.
        n_channels (int): Number of channels.
        steps (int): Number of Gibbs' sampling steps.
        use_gpu (boolean): Whether GPU should be used or not.
        batch_size (int): Amount of samples per batch.
        epochs (int): Number of training epochs.
        val (torchtext.data.Dataset): Training dataset.
        val (torchtext.data.Dataset): Validation dataset.

    """

    def f(p):
        """Instantiates a model, gathers variables from meta-heuritics, trains and evaluates over validation data.

        Args:
            p (float): Array of variables/parameters.

        Returns:
            Mean squared error (MSE) of validation set.

        """

        # Optimization parameters
        filter_shape = (int(p[0][0]), int(p[0][0]))
        n_filters = int(p[1][0])
        lr = p[2][0]
        momentum = p[3][0]
        decay = p[4][0]

        # Initializes the model
        model = ConvRBM(visible_shape, filter_shape, n_filters, n_channels, steps, lr, momentum, decay, use_gpu)

        # Trains the model using the training set
        model.fit(train, batch_size, epochs)

        # Reconstructs over the validation set
        mse, _ = model.reconstruct(val)

        return mse.item()

    return f
