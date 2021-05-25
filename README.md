# anytime: robust learning anytime

In this repository, we provide software and demonstrations related to the following paper:

- <a href="https://arxiv.org/pdf/2105.11135.pdf">Robust learning with anytime-guaranteed feedback</a>. Matthew J. Holland. *Preprint.*

This repository contains code which can be used to faithfully reproduce all the experimental results given in the above paper, and it can be easily applied to more general machine learning tasks outside the examples considered here.


A table of contents for this README file:

- <a href="#setup_init">Setup: initial software preparation</a>
- <a href="#setup_data">Setup: preparing the benchmark data sets</a>
- <a href="#start">Getting started</a>
- <a href="#demos">Demos and visualization</a>
- <a href="#safehash">Safe hash value</a>


<a id="setup_init"></a>
## Setup: initial software preparation

To begin, please ensure you have the <a href="https://github.com/feedbackward/mml#prerequisites">prerequisite software</a> used in the setup of our `mml` repository.

Next, make a local copy of the repository and create a virtual environment for working in as follows:

```
$ git clone https://github.com/feedbackward/mml.git
$ git clone https://github.com/feedbackward/anytime.git
$ conda create -n anytime python=3.9 jupyter matplotlib pip pytables scipy
$ conda activate anytime
```

Having made (and activated) this new environment, we would like to use `pip` to install the supporting libraries for convenient access. This is done easily, by simply running

```
(anytime) cd [mml path]/mml
(anytime) pip install -e ./
```

with the `[mml path]` placeholder replaced with the path to wherever you placed the repositories. If you desire a safe, tested version of `mml`, just run

```
(anytime) $ git checkout [safe hash mml]
```

and then do the `pip install -e ./` command mentioned above. The `[safe hash mml]` placeholder is to be replaced using the safe hash value given at the end of this document.


<a id="setup_data"></a>
## Setup: preparing the benchmark data sets

Please follow the instructions under <a href="https://github.com/feedbackward/mml#data">"Acquiring benchmark datasets"</a> using our `mml` repository. The rest of this README assumes that the user has prepared any desired benchmark datasets, stored in a local data storage directory (default path is `[path to mml]/mml/mml/data` as specified by the variable `dir_data_towrite` in `mml/mml/config.py`.

One __important__ step is to ensure that once you've acquired the benchmark data using `mml`, you must ensure that `anytime` knows where that data is. To do this, set `dir_data_toread` in `setup_data.py` to the directory housing the HDF5 format data sub-directories (default setting: your home directory).


<a id="start"></a>
## Getting started

We have basically three types of files:

- __Setup files:__ these take the form `setup_*.py`.
  - Configuration for all elements of the learning process, with one setup file for each of the following major categories: learning algorithms, data preparation, learned model evaluation, loss functions, models, result processing, and general-purpose training functions.

- __Driver scripts:__ just one at present, called `learn_driver.py`.
  - This script controls the flow of the learning procedure and handle all the clerical tasks such as organizing, naming, and writing numerical results to disk. No direct modification to this file is needed to run the experiments in the above paper.

- __Execution scripts:__ just one at present, called `learn_run.sh`.
  - The choice of algorithm, model, data generation protocol, among other key parameters is made within this simple shell script. Basically, parameters are specified explicitly, and these are then passed to the driver script as options.

The experiments using real-world datasets require the user to run the driver script themselves; this is described in more detail within the demo notebook.


<a id="demos"></a>
## List of demos

This repository includes detailed demonstrations to walk the user through re-creating the results in the paper cited at the top of this document. Below is a list of demo links which give our demos (constructed in Jupyter notebook form) rendered using the useful <a href="https://github.com/jupyter/nbviewer">nbviewer</a> service.

- <a href="https://nbviewer.jupyter.org/github/feedbackward/anytime/blob/main/anytime/demo.ipynb">Empirical analysis of "anytime" feedback</a>


<a id="safehash"></a>
## Safe hash value

- Replace `[safe hash mml]` with `9eca553bcd2c0977df59426c1ddf89b07808aebc`.

__Date of safe hash test:__ 2021/02/17.
