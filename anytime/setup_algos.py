'''Setup: algorithms.'''

## External modules.
import numpy as np

## Internal modules.
from mml.algos import Algorithm
from mml.algos.gd import GD_ERM
from mml.algos.linesearch import LineSearch


###############################################################################


class Weighted_Average(Algorithm):
    '''
    An Algorithm for sequentially computing
    the simplest possible sequential weighted
    average of parameter candidates.
    '''
    
    def __init__(self, model_main=None, model_ancillary=None):
        super().__init__(model=model_main, loss=None, name=None)
        self.model_ancillary = model_ancillary
        self.weight_sum = 1.0
        return None


    def update(self, X=None, y=None):
        for pn, p in self.paras.items():
            p *= self.weight_sum
            p += self.model_ancillary.paras[pn]
            p /= self.weight_sum + 1.0
        self.weight_sum += 1.0
        return None


class GD_ERM_Anytime(LineSearch):
    '''
    "Anytime" variant of GD_ERM.
    This algorithm is resposible for updating
    the so-called "ancillary" sequence.
    '''

    def __init__(self, model_query=None, step_coef=None,
                 model=None, loss=None, name=None):
        super().__init__(model=model, loss=loss, name=name)
        self.step_coef = {}
        for pn, p in self.paras.items():
            self.step_coef[pn] = step_coef
        self.model_query = model_query
        return None

    
    def newdir(self, X=None, y=None):
        loss_grads = self.loss.grad(model=self.model_query, X=X, y=y)
        newdirs = {}
        for pn, g in loss_grads.items():
            newdirs[pn] = -g.mean(axis=0, keepdims=False)
        return newdirs


    def stepsize(self, newdirs=None, X=None, y=None):
        '''
        Just return the pre-fixed step sizes.
        '''
        return self.step_coef


class GD_Robust_Anytime(LineSearch):
    '''
    A robust anytime GD procedure.
    '''
    
    def __init__(self, conf, num_data, smooth,
                 anchor_primal, anchor_dual,
                 model_query=None, step_coef=None,
                 model=None, loss=None, name=None):
        super().__init__(model=model, loss=loss, name=name)
        self.smooth = smooth
        self.anchor_primal = anchor_primal
        self.anchor_dual = anchor_dual
        self.thres = np.sqrt( num_data / np.log(1.0/conf) )
        self.step_coef = {}
        for pn, p in self.paras.items():
            self.step_coef[pn] = step_coef
        self.model_query = model_query
        return None


    def newdir(self, X=None, y=None):

        loss_grads = self.loss.grad(model=self.model_query,
                                    X=X, y=y)
        newdirs = {}
        for pn, g in loss_grads.items():

            ## Unbiased gradient (works fine with mini-batches >= 1).
            g_raw = g.mean(axis=0, keepdims=False)

            ## Order of the norm.
            norm_ord = 2 if g_raw.ndim != 2 else "fro"
        
            ## Distance from anchor in dual space.
            dist_dual = np.linalg.norm(
                x=g_raw-self.anchor_dual[pn],
                ord=norm_ord
            )
        
            ## Distance from anchor in primal space.
            dist_primal = np.linalg.norm(
                x=self.model_query.paras[pn]-self.anchor_primal[pn],
                ord=norm_ord
            )
            
            ## Final thresholding check.
            if dist_dual <= self.smooth*dist_primal + self.thres:
                newdirs[pn] = -g_raw
            else:
                newdirs[pn] = -self.anchor_dual[pn]

        ## Having obtained new directions for all parameters, return.
        return newdirs


    def stepsize(self, newdirs=None, X=None, y=None):
        '''
        Just return the pre-fixed step sizes.
        '''
        return self.step_coef


## Simple parser for algorithm objects.

def get_algo(name_main, name_ancillary,
             model_main, model_ancillary,
             loss, **kwargs):

    if name_main == "Ave":
        algo_main = Weighted_Average(model_main=model_main,
                                     model_ancillary=model_ancillary)
    else:
        algo_main = None
    
    if name_ancillary == "SGD":
        algo_ancillary = GD_ERM(
            step_coef=kwargs["step_size"],
            model=model_ancillary,
            loss=loss
        )
    elif name_ancillary == "Anytime-SGD":
        algo_ancillary = GD_ERM_Anytime(
            model_query=model_main,
            step_coef=kwargs["step_size"],
            model=model_ancillary,
            loss=loss
        )
    elif name_ancillary == "Anytime-Robust-SGD":
        algo_ancillary = GD_Robust_Anytime(
            conf=0.05, # hard-coded.
            num_data=kwargs["num_data"],
            smooth=1.0, # hard-coded.
            anchor_primal=kwargs["anchor_primal"],
            anchor_dual=kwargs["anchor_dual"],
            model_query=model_main,
            step_coef=kwargs["step_size"],
            model=model_ancillary,
            loss=loss
        )
    else:
        algo_ancillary = None

    return (algo_main, algo_ancillary)


###############################################################################
