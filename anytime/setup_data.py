'''Setup: preparation of data sets.'''

## External modules.
import numpy as np
import os
from pathlib import Path
from tables import open_file

## Internal modules.
from mml.data import dataset_dict, get_data_general
from mml.models.linreg import LinearRegression
from mml.utils.rgen import get_generator, get_stats


###############################################################################


## If benchmark data is to be used, specify the directory here.
dir_data_toread = os.path.join(str(Path.home()),
                               "mml", "mml", "data")


## First set dataset parameter dictionary with standard values
## for all the benchmark datasets in mml.
dataset_paras = dataset_dict


## Data generation procedure.

def get_data(dataset, rg=None):
    '''
    Takes a string, return a tuple of data and parameters.
    '''
    if dataset in dataset_paras:
        return get_data_general(dataset=dataset,
                                paras=dataset_paras[dataset],
                                rg=rg,
                                directory=dir_data_toread)
    else:
        raise ValueError(
            "Did not recognize dataset {}.".format(dataset)
        )


###############################################################################
