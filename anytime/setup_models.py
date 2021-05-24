'''Setup: models.'''

## Internal modules.
from mml.models.linreg import LinearRegression, LinearRegression_Multi


###############################################################################


## The main parser function, returning model instances.

def get_model(name, paras_init=None, rg=None, **kwargs):

    if name == "linreg_multi":
        return LinearRegression_Multi(num_features=kwargs["num_features"],
                                      num_outputs=kwargs["num_classes"],
                                      paras_init=paras_init, rg=rg)
    elif name == "linreg":
        return LinearRegression(num_features=kwargs["num_features"],
                                paras_init=paras_init, rg=rg)
    else:
        raise ValueError("Please pass a valid model name.")
    

###############################################################################
