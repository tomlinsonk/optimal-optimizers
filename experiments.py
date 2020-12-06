import itertools
import pickle
from functools import partial
from multiprocessing import Pool
from scipy.stats import loguniform, uniform, rv_continuous


import torch
from scipy.stats import chi2
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

from choice_models import fit, Logit, CDM, ConditionalLogit, LCL
from datasets import SFShop, SFWork, Expedia, Sushi, FeatureSushi, Allstate

N_THREADS = 30

TRIALS = 32

RESULTS_DIR = 'results'


OPTIMIZERS = {
    'Adadelta': {'constructor': torch.optim.Adadelta, 'params': {'lr': loguniform(10**-2, 10**2), 'rho': uniform(0.8, 0.2)}},
    'Adagrad': {'constructor': torch.optim.Adagrad, 'params': {'lr': loguniform(10**-4, 10**0)}},
    'Adam': {'constructor': torch.optim.Adam, 'params': {'lr': loguniform(10**-5, 10**-1)}},
    'AMSGrad Adam': {'constructor': torch.optim.Adam, 'params': {'lr': loguniform(10 ** -5, 10 ** -1), 'amsgrad': True}},
    'Adamax': {'constructor': torch.optim.Adamax, 'params': {'lr': loguniform(10**-5, 10**-1)}},
    'ASGD': {'constructor': torch.optim.ASGD, 'params': {'lr': loguniform(10**-4, 10**0), 'lambd': loguniform(10**-6, 10**-2), 'alpha': uniform(0.5, 0.5), 't0': loguniform(10**4, 10**8)}},
    'LBFGS': {'constructor': torch.optim.LBFGS, 'params': {'lr': loguniform(10**-2, 10**2)}},
    'RMSprop': {'constructor': torch.optim.RMSprop, 'params': {'lr': loguniform(10**-4, 10**0), 'alpha': uniform(0.8, 0.2)}},
    'Rprop': {'constructor': torch.optim.Rprop, 'params': {'lr': loguniform(10**-4, 10**0)}},
    'SGD': {'constructor': torch.optim.SGD, 'params': {'lr': loguniform(10**-4, 10**0)}},
    'Nesterov SGD': {'constructor': torch.optim.SGD, 'params': {'lr': loguniform(10**-4, 10**0), 'nesterov': True}},
    'Momentum SGD': {'constructor': torch.optim.SGD, 'params': {'lr': loguniform(10**-4, 10**0), 'momentum': loguniform(10**-5, 10**-1)}},
}


def get_optimizer(optim_name, model):
    constructor = OPTIMIZERS[optim_name]['constructor']
    params = {key: value.rvs() if not isinstance(value, bool) else value
              for key, value in OPTIMIZERS[optim_name]['params'].items()}

    return constructor(model.parameters(), **params)


def item_identity_dataset_helper(args):
    dataset, optim_name, Model, seed = args
    np.random.seed(seed)

    choice_sets, choices, person_df = dataset.load_pytorch()
    n_items = choice_sets.size(1)
    model = Model(n_items)
    optimizer = get_optimizer(optim_name, model)

    results = fit(model, (choice_sets, choices), optimizer=optimizer, show_progress=False)

    return args, results


def item_feature_dataset_helper(args):
    dataset, optim_name, Model, seed = args
    np.random.seed(seed)

    choice_set_features, choice_set_lengths, choices, person_df = dataset.load_pytorch()
    n_features = choice_set_features.size(-1)
    model = Model(n_features)
    optimizer = get_optimizer(optim_name, model)

    results = fit(model, (choice_set_features, choice_set_lengths, choices), optimizer=optimizer, show_progress=False)

    return args, results


def compare_item_identity_models_experiment():
    datasets = [SFWork, SFShop, Sushi]
    model_classes = [Logit, CDM]
    seeds = range(TRIALS)

    params = list(itertools.product(datasets, OPTIMIZERS.keys(), model_classes, seeds))

    all_results = dict()

    with Pool(N_THREADS) as pool:
        for args, results in tqdm(pool.imap_unordered(item_identity_dataset_helper, params), total=len(params)):
            all_results[args] = results

    with open(f'{RESULTS_DIR}/item_identity_results.pt', 'wb') as f:
        torch.save((datasets, model_classes, seeds, OPTIMIZERS.keys(), all_results), f)


def compare_item_feature_models_experiment():
    datasets = [Expedia, Allstate, FeatureSushi]
    model_classes = [ConditionalLogit, LCL]
    seeds = range(TRIALS)

    params = list(itertools.product(datasets, OPTIMIZERS.keys(), model_classes, seeds))

    all_results = dict()

    with Pool(N_THREADS) as pool:
        for args, results in tqdm(pool.imap_unordered(item_feature_dataset_helper, params), total=len(params)):
            all_results[args] = results

    with open(f'{RESULTS_DIR}/item_feature_results.pt', 'wb') as f:
        torch.save((datasets, model_classes, seeds, OPTIMIZERS.keys(), all_results), f)


if __name__ == '__main__':
    compare_item_identity_models_experiment()
    compare_item_feature_models_experiment()
