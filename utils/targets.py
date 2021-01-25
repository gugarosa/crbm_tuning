import utils.objects as o


def fine_tune_reconstruction(model_str, n_input, n_hidden, steps, lr, momentum, decay, T, use_gpu, train, val):
    """Wraps the reconstruction task for optimization purposes.

    Args:
        model (RBM): Child object from RBM class.
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

        # Instantiates the model
        model_obj = o.get_model(model_str).obj
        model =  model_obj(n_visible=n_input, n_hidden=n_hidden, steps=steps, learning_rate=lr,
                           momentum=momentum, decay=decay, temperature=T, use_gpu=use_gpu)

        # Trains the model over the training set
        model.fit(train, batch_size=128, epochs=5)

        # Reconstructs over validation set
        mse, _ = model.reconstruct(val)

        return mse.item()

    return f
