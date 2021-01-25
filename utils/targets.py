import utils.objects as o


def reconstruction(model_str, n_input, visible_shape, n_hidden, filter_shape, n_filters, n_channels,
                   steps, lr, momentum, decay, T, use_gpu, batch_size, epochs, train, val):
    """Wraps the reconstruction task for optimization purposes.

    Args:
        model_str (str): String holding the RBM-based class.
        n_input (int): Amount of visible units.
        visible_shape (tuple): Shape of visible units.
        n_hidden (int): Amount of hidden units.
        filter_shape (tuple): Shape of filters.
        n_filters (int): Number of filters.
        n_channels (int): Number of channels.
        steps (int): Number of Gibbs' sampling steps.
        learning_rate (float): Learning rate.
        momentum (float): Momentum parameter.
        decay (float): Weight decay used for penalization.
        T (float): Temperature factor.
        use_gpu (boolean): Whether GPU should be used or not.
        batch_size (int): Amount of samples per batch.
        epochs (int): Number of training epochs.
        val (torchtext.data.Dataset): Training dataset.
        val (torchtext.data.Dataset): Validation dataset.

    """

    def f(w):
        """Instantiates a model, gathers variables from meta-heuritics, trains and evaluates over validation data.

        Args:
            p (float): Array of variables/parameters.

        Returns:
            Mean squared error (MSE) of validation set.

        """

        # Gets the model
        model_obj = o.get_model(model_str).obj

        # Checks if the model is supposed to use RBM arguments
        if model_str == 'rbm':
            # Instantiates the corresponding model
            model = model_obj(n_visible=n_input, n_hidden=n_hidden, steps=steps, learning_rate=lr,
                              momentum=momentum, decay=decay, temperature=T, use_gpu=use_gpu)

        # Checks if the model is supposed to use ConvRBM arguments
        elif model_str == 'crbm':
            # Instantiates the corresponding model
            model = model_obj(visible_shape=visible_shape, filter_shape=filter_shape, n_filters=n_filters,
                              n_channels=n_channels, steps=steps, learning_rate=lr,
                              momentum=momentum, decay=decay, use_gpu=use_gpu)

        # Trains the model using the training set
        model.fit(train, batch_size=batch_size, epochs=epochs)

        # Reconstructs over the validation set
        mse, _ = model.reconstruct(val)

        return mse.item()

    return f
