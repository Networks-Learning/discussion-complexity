# Crowdjudged relationships

This is the current directory structure with short descriptions:

▾ cjr/                         | Module containing all the code
  ▾ models/
      consts.py*               | Soft-link for easy importing to ../utils/utils.py
      lin_thres.py             | Model for the 1D model
      senti_adjust.py          | Not used.
      unif.py                  | Model which assumes uniform opinion distribution
  ▾ synth/                     | Synthetic data generation
      __init__.py
      consts.py*               | Soft-link for easy importing to ../utils/utils.py
      generate_synth_data.py   | Generator of synthetic data (executable)
  ▾ utils/
      __init__.py
      consts.py                | String constants
      utils.py                 | Common utility functions
    __init__.py
▸ data/                        | Synthetic/real/sample datasets
▾ notebooks/                   | These are sample notebooks.
    DCCP experiment.ipynb      | Example DCCP problems/constraints (to be ignored)
    Sentir model.ipynb         | Proof of concept with 2D embedding.
    Synthetic-data-[...].ipynb | Experiments with 1D embedding.
  README.md                    | This file.

## Running notebooks

The notebooks included are just samples. The recommended way to run them is to
create a copy each and then run them while keeping CWD as `./notebooks`.

Please do not overwrite the notebooks in commits; those merge conflicts are
very painful to resolve.

## Requirements

These python packages are required:

  - `click`
  - `cvxpy`
  - `dccp `
  - `numpy`
  - `decorated_options` (`pip install decorated_options`)
  - `seaborn`

All of these can be installed using `pip` while some of them are available on
`conda` (preferred).
