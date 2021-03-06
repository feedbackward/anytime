'''Driver script for tests of anytime SGD and related methods.'''

## External modules.
import argparse
from copy import deepcopy
import json
import numpy as np
import os

## Internal modules.
from mml.utils import makedir_safe
from setup_algos import get_algo
from setup_data import get_data
from setup_eval import get_eval, eval_model, eval_write
from setup_losses import get_loss
from setup_models import get_model
from setup_results import results_dir
from setup_train import train_epoch


###############################################################################


## Basic setup.

parser = argparse.ArgumentParser(description="Arguments for driver script.")

parser.add_argument("--algo-ancillary",
                    help="Ancillary algorithm class. (default: SGD)",
                    type=str, default="SGD", metavar="S")
parser.add_argument("--algo-main",
                    help="Main algorithm class. (default: None)",
                    type=str, default=None, metavar="S")
parser.add_argument("--batch-size",
                    help="Mini-batch size for algorithms (default: 1).",
                    type=int, default=1, metavar="N")
parser.add_argument("--data",
                    help="Specify data set to be used (default: None).",
                    type=str, default=None, metavar="S")
parser.add_argument("--entropy",
                    help="For data-gen seed sequence (default is random).",
                    type=int,
                    default=np.random.SeedSequence().entropy,
                    metavar="N")
parser.add_argument("--loss",
                    help="Loss name. (default: quadratic)",
                    type=str, default="quadratic", metavar="S")
parser.add_argument("--model",
                    help="Model class. (default: linreg)",
                    type=str, default="linreg", metavar="S")
parser.add_argument("--num-epochs",
                    help="Number of epochs to run (default: 3)",
                    type=int, default=3, metavar="N")
parser.add_argument("--num-trials",
                    help="Number of independent random trials (default: 1)",
                    type=int, default=1, metavar="N")
parser.add_argument("--step-size",
                    help="Step size parameter (default: 0.01)",
                    type=float, default=0.01, metavar="F")
parser.add_argument("--task-name",
                    help="A task name. Default is the word default.",
                    type=str, default="default", metavar="S")

## Parse the arguments passed via command line.
args = parser.parse_args()
if args.data is None:
    raise TypeError("Given --data=None, should be a string.")

## Name to be used identifying the results etc. of this experiment.
towrite_name = args.task_name+"-"+"_".join([args.model,
                                            args.algo_ancillary])
if len(args.algo_main) > 0:
    towrite_name += "_{}".format(args.algo_main)

## Prepare a directory to save results.
towrite_dir = os.path.join(results_dir, args.data)
makedir_safe(towrite_dir)

## Setup of manually-specified seed sequence for data generation.
ss_parent = np.random.SeedSequence(args.entropy)
ss_children = ss_parent.spawn(args.num_trials)
rg_children = [np.random.default_rng(seed=ss) for ss in ss_children]

## Main process.
if __name__ == "__main__":
    
    ## Prepare the loss for training.
    loss = get_loss(name=args.loss)

    ## Arguments for algorithms.
    algo_kwargs = {}
    
    ## Arguments for models.
    model_kwargs = {}
    
    ## Start the loop over independent trials.
    for trial in range(args.num_trials):

        ## Get trial-specific random generator.
        rg_child = rg_children[trial]
        
        ## Load in data.
        print("Doing data prep.")
        (X_train, y_train, X_val, y_val,
         X_test, y_test, ds_paras) = get_data(dataset=args.data, rg=rg_child)

        ## Data index.
        data_idx = np.arange(len(X_train))
        
        ## Prepare evaluation metric(s).
        eval_dict = get_eval(loss_name=args.loss,
                             model_name=args.model, **ds_paras)
        
        ## Model setup.
        model_ancillary = get_model(name=args.model,
                                    paras_init=None,
                                    rg=rg_child, **model_kwargs, **ds_paras)
        model_main = get_model(name=args.model,
                               paras_init=deepcopy(model_ancillary.paras),
                               rg=rg_child, **model_kwargs, **ds_paras)
        model_anchor = get_model(name=args.model,
                                 paras_init=deepcopy(model_ancillary.paras),
                                 rg=rg_child, **model_kwargs, **ds_paras)
        
        ## Prepare algorithms.
        loss_grads = loss.grad(model=model_anchor, X=X_train, y=y_train)
        anchor_dual = {}
        for pn, g in loss_grads.items():
            anchor_dual[pn] = g.mean(axis=0, keepdims=False)
        del loss_grads
        algo_kwargs.update(
            {"num_data": len(X_train),
             "anchor_primal": model_anchor.paras,
             "anchor_dual": anchor_dual,
             "step_size": args.step_size*2/np.sqrt(len(X_train))}
        )
        algo_main, algo_ancillary = get_algo(
            name_main=args.algo_main,
            name_ancillary=args.algo_ancillary,
            model_main=model_main,
            model_ancillary=model_ancillary,
            loss=loss,
            **ds_paras, **algo_kwargs
        )
        
        ## Prepare storage for performance evaluation this trial.
        store_train = {
            key: np.zeros(shape=(args.num_epochs,1),
                          dtype=np.float32) for key in eval_dict.keys()
        }
        if X_test is not None:
            store_test = {
                key: np.zeros_like(
                    store_train[key]
                ) for key in eval_dict.keys()
            }
        else:
            store_test = {}
        storage = (store_train, store_test)
        
        ## Loop over epochs, done in the parent process.
        for epoch in range(args.num_epochs):
            
            print("(Tr {}) Ep {} starting.".format(trial, epoch))
            
            ## Shuffle data.
            rg_child.shuffle(data_idx)
            X_train = X_train[data_idx,:]
            y_train = y_train[data_idx,:]

            ## Carry out one epoch's worth of training.
            train_epoch(algo_main=algo_main,
                        algo_ancillary=algo_ancillary,
                        loss=loss,
                        X=X_train,
                        y=y_train,
                        batch_size=args.batch_size)

            ## Evaluate performance of the sub-process candidates.
            if algo_main is None:
                eval_model(epoch=epoch,
                           model=model_ancillary,
                           storage=storage,
                           data=(X_train,y_train,X_test,y_test),
                           eval_dict=eval_dict)
            else:
                eval_model(epoch=epoch,
                           model=model_main,
                           storage=storage,
                           data=(X_train,y_train,X_test,y_test),
                           eval_dict=eval_dict)
            
            print("(Tr {}) Ep {} finished.".format(trial, epoch), "\n")


        ## Write performance for this trial to disk.
        perf_fname = os.path.join(towrite_dir,
                                  towrite_name+"-"+str(trial))
        eval_write(fname=perf_fname,
                   storage=storage)

    ## Write a JSON file to disk that summarizes key experiment parameters.
    dict_to_json = vars(args)
    dict_to_json.update({
        "entropy": args.entropy
    })
    towrite_json = os.path.join(towrite_dir, towrite_name+".json")
    with open(towrite_json, "w", encoding="utf-8") as f:
        json.dump(obj=dict_to_json, fp=f,
                  ensure_ascii=False,
                  sort_keys=True, indent=4)


###############################################################################
