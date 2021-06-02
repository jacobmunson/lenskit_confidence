

# Notice

Our work takes the Lenskit for Python package and modifies certains function to support confidence-aware approaches. The LKPY package is authored and maintained by Michael Ekstrand (available: https://github.com/lenskit) and the contents herein do not reflect on the Lenskit project beyond the scope of this work. This work is not intended to join the larger Lenskit ecosystem and it will not necessarily be maintained/updated as new versions of Lenskit come out. 

- Note: Most directories have not been edited; please advise that most readme pages are original from the LKPY github
- Note: installing the original Lenskit package WILL NOT offer the functionality discussed in the paper; installation must be done locally as described below

## Installation Instructions

### Get local copy
- Clone Github repository
- On my machine, I put it into "Documents/Github/lenskit_confidence"
- Resource on local installations: https://www.depts.ttu.edu/hpcc/userguides/application_guides/python.packages.local_installation.php

### Install local copy
- Open terminal
- Navigate to "Documents/Github/lenskit_confidence"
- Run "python setup.py install"
- Note: C++/other dependencies may be required
- Optional: setup a Conda environment for easy package management

### Running notebooks
- Open Anaconda Navigator and/or Jupyter Notebook
- Navigate to "Documents/Github/lenskit_confidence/examples"
- Pick a notebook to run
    - "BaselinesEval_UBCF_IBCF.ipynb" for baseline models walkthrough
    - "Confidence-aware-UBCF-MultiEval.ipynb" for Confidence-aware UBCF walkthrough
    - "Confidence-aware-IBCF-MultiEval.ipynb" for Confidence-aware IBCF walkthrough
    - By default, model output is say in 'my-eval' folder
- Analyze results
    - Some analysis is detailed in the respective notebooks
    - More involved analysis is done in independent notebooks
            - "TopN Evaluation - Baselines-loop.ipynb" is a notebook for looking through list sizes and running Precision and NDCG evaluations with the baseline models
            - "TopN Evaluation - Confidence-aware-loop.ipynb" is a notebook for looking through list sizes and running Precision and NDCG evaluations with the confidence-aware models
            - Note: make sure the intended folder exists

### Some common hangups
- Make sure "binpickle" is installed and updated: conda install -c conda-forge binpickle
    - Gives error: "TypeError: __new__() keywords must be strings"
- Make sure "pyarrow" is installed and updated: conda install pyarrow
    - Gives error: "TypeError: __cinit__() got an unexpected keyword argument 'index'"
- Make sure "numba" is installed: conda install numba, conda install numba-scipy
- Make sure "tqdm" is installed for in-notebook updates
- Make sure "pickle5" is updated: conda install -c conda-forge pickle5
        - Gives error: module 'pickle5' has no attribute 'PickleBuffer'

### Some data commentary
- This paper used the MovieLens 1M, 10M, 20M datasets and the Jester 4.1M dataset
- MovieLens is available here: https://grouplens.org/datasets/movielens/
        - MovieLens can be pulled into the Lenskit framework by existing function (as demonstrated in the notebooks)
- Jester is available here: http://eigentaste.berkeley.edu/dataset/
        - The Jester dataset requires some cleaning
        - The script to do so is "lenskit_confidence/examples/jester_data_cleaning.R"
- The graphics in the paper were produced with ggplot: the scripts to produce the graphics are in "lenskit_confidence/examples/" under various "TopN_xxxxxxx_graphics.R" type names

### Everything below is original to LKPY and has not been edited 

# Python recommendation tools

![Test Suite](https://github.com/lenskit/lkpy/workflows/Test%20Suite/badge.svg)
[![codecov](https://codecov.io/gh/lenskit/lkpy/branch/master/graph/badge.svg)](https://codecov.io/gh/lenskit/lkpy)
[![Maintainability](https://api.codeclimate.com/v1/badges/c02098c161112e19c148/maintainability)](https://codeclimate.com/github/lenskit/lkpy/maintainability)

LensKit is a set of Python tools for experimenting with and studying recommender
systems.  It provides support for training, running, and evaluating recommender
algorithms in a flexible fashion suitable for research and education.

LensKit for Python (LKPY) is the successor to the Java-based LensKit project.

If you use LensKit for Python in published research, please cite:

> Michael D. Ekstrand. 2020.
> LensKit for Python: Next-Generation Software for Recommender Systems Experiments.
> In <cite>Proceedings of the 29th ACM International Conference on Information and Knowledge Management</cite> (CIKM '20).
> DOI:[10.1145/3340531.3412778](https://dx.doi.org/10.1145/3340531.3412778>).
> arXiv:[1809.03125](https://arxiv.org/abs/1809.03125) [cs.IR].

## Installing

To install the current release with Anaconda (recommended):

    conda install -c lenskit lenskit

Or you can use `pip`:

    pip install lenskit

To use the latest development version, install directly from GitHub:

    pip install -U git+https://github.com/lenskit/lkpy

Then see [Getting Started](https://lkpy.lenskit.org/en/latest/GettingStarted.html)

## Developing

[issues]: https://github.com/lenskit/lkpy/issues
[workflow]: https://github.com/lenskit/lkpy/wiki/DevWorkflow

To contribute to LensKit, clone or fork the repository, get to work, and submit
a pull request.  We welcome contributions from anyone; if you are looking for a
place to get started, see the [issue tracker][].

Our development workflow is documented in [the wiki][workflow]; the wiki also
contains other information on *developing* LensKit. User-facing documentation is
at <https://lkpy.lenskit.org>.


We recommend using an Anaconda environment for developing LensKit.  To set this
up, run:

    python setup.py dep_info --conda-environment dev-env.yml
    conda env create -f dev-env.yml

This will create a Conda environment called `lkpy-dev` with the packages
required to develop and test LensKit.

We don't maintain the Conda environment specification directly - instead, we
maintain information in `setup.cfg` to be able to generate it, so that we define
dependencies and versions in one place (well, two, if you count the `meta.yaml`
file used to build the Conda recipes).  The `dep_info` setuptools command will
generate a Conda environment specification from the current dependencies in
`setup.cfg`.

## Resources

- [Documentation](https://lkpy.lenskit.org)
- [Mailing list, etc.](https://lenskit.org/connect)

## Acknowledgements

This material is based upon work supported by the National Science Foundation
under Grant No. IIS 17-51278. Any opinions, findings, and conclusions or
recommendations expressed in this material are those of the author(s) and do not
necessarily reflect the views of the National Science Foundation.
