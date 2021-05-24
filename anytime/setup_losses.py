'''Setup: loss functions used for training and evaluation.'''

## Internal modules.
from mml.losses.classification import Zero_One
from mml.losses.logistic import Logistic
from mml.losses.quadratic import Quadratic


###############################################################################


## A dictionary of instantiated losses.

dict_losses = {
    "logistic": Logistic(),
    "quadratic": Quadratic(),
    "zero_one": Zero_One()
}

def get_loss(name):
    '''
    A simple parser that returns a loss instance.
    '''
    return dict_losses[name]


###############################################################################
