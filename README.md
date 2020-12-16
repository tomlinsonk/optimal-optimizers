# CS 6787 Final Project: Optimal Optimizers for Learning Discrete Choice Models

This repository contains code for my [CS 6787: Advanced ML Systems](https://www.cs.cornell.edu/courses/cs6787/2020fa/) final project. The report can be found in `report/`.

## Libraries
- Python 3.8.5
  - matplotlib 3.3.0
  - numpy 1.19.1
  - scipy 1.5.1
  - tqdm 4.48.0
  - torch 1.6.0
  - pandas 1.0.5
  
## Files
- `choice_models.py`: implementations of the logit, CDM, conditional logit, and LCL models and training procedure
- `datasets.py`: dataset processing and management
- `experiments.py`: parallelized optimizer tests
- `plot.py`: makes plots and prints tables

The `plots/` directory containes all loss trajectory plots and the `results/` directory contains saved results from my run of the experiments.

The CDM implementation was adapted from [Arjun Seshadri's CDM code](https://github.com/arjunsesh/cdm-icml), which accompanies the paper
> Arjun Seshadri, Alex Peysakhovich, and Johan Ugander. Discovering Context Effects from Raw Choice Data. In International Conference on Machine Learning, 2019. 5660–5669.

The LCL implementation was adapted from [my LCL code](https://github.com/tomlinsonk/feature-context-effects), which accompanies the preprint
> Kiran Tomlinson and Austin R. Benson. Learning Interpretable Feature Context Effects in Discrete Choice. https://arxiv.org/abs/2009.03417, 2020.

## Data
First, create a directory to hold the data (I'll refer to this directory as `data/`, but you can name it anything). Create a subdirectory of `data/` called `pickles/`,
which will hold processed versions of the larger datasets. Download the SFWork/SFShop data from this [Google Drive link](https://drive.google.com/file/d/15CMJ7_caeKXcXkMIRGVWSnp5M18S8T6G/view?usp=sharing).
Place the unzipped `SF-raw/` directory in a new dirctory `data/SF/`. Download the Sushi dataset [here](http://www.kamishima.net/sushi/) (the file `sushi3-2016.zip`).
Place the unzipped `sushi3-2016/` directory in `data/`. Finally, the [Expedia](https://www.kaggle.com/c/expedia-personalized-sort/data) and [Allstate](https://www.kaggle.com/c/allstate-purchase-prediction-challenge)
datasets are available directly from Kaggle. Place both the `expedia-personalized-sort` and `allstate-purchase-prediction-challenge/` directories in `data/`. Finally, create a file called `config.yml` in `cs6787-final-project/` with a single line, replacing the path with yours:

`datadir: '/path/to/data/'`

Now you're ready to run the experiments!

## Reproducibility
To create the plots and tables from the data provided in the repository, just run `python3 plot.py`. To run all experiments, 
run `python3 experiments.py` after downloading the datasets as described above. By default, `experiments.py` uses 30 threads--you may wish to modify this by changing the 
`N_THREADS` constant.
