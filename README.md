# Crowdjudged relationships

This is the current directory structure with short descriptions:


    ▾ code/                         | All Code
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
    README.md                       | This file.

## Requirements

These python packages are required:

  - `click`
  - `cvxpy`
  - `dccp `
  - `numpy`
  - `decorated_options` (`pip install decorated_options`)
  - `seaborn`
  - `z3-solver`

All of these can be installed using `pip` while some of them are available on
`conda` (preferred).
