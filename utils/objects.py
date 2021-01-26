from opytimizer.optimizers.evolutionary import ga, gp


class MetaHeuristic:
    """A MetaHeuristic class to help users in selecting distinct meta-heuristics from the command line.

    """

    def __init__(self, obj, hyperparams):
        """Initialization method.

        Args:
            obj (Optimizer): An Optimizer-child instance.
            hyperparams (dict): Meta-heuristic hyperparams.

        """

        # Creates a property to hold the class itself
        self.obj = obj

        # Creates a property to hold the hyperparams
        self.hyperparams = hyperparams


# Defines a meta-heuristic dictionary constant with the possible values
MH = dict(
    ga=MetaHeuristic(ga.GA, dict(p_selection=0.75, p_mutation=0.25, p_crossover=0.5)),
    gp=MetaHeuristic(gp.GP, dict(p_reproduction=0.25, p_mutation=0.1, p_crossover=0.2, prunning_ratio=0.0))
)


def get_mh(name):
    """Gets a meta-heuristic by its identifier.

    Args:
        name (str): Meta-heuristic's identifier.

    Returns:
        An instance of the MetaHeuristic class.

    """

    # Tries to invoke the method
    try:
        # Returns the corresponding object
        return MH[name]

    # If object is not found
    except:
        # Raises a RuntimeError
        raise RuntimeError(f'Meta-heuristic {name} has not been specified yet.')
