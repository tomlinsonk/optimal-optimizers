import itertools

import torch
import numpy as np
import matplotlib.pyplot as plt

from choice_models import LCL
from datasets import Expedia


def plot(results_fname):

    datasets, model_classes, seeds, optimizers, results = torch.load(results_fname)


    losses = {(a, b, c): [] for a in model_classes for b in datasets for c in optimizers}
    run_times = {(a, b, c): [] for a in model_classes for b in datasets for c in optimizers}
    gradients = {(a, b, c): [] for a in model_classes for b in datasets for c in optimizers}
    iterations = {(a, b, c): [] for a in model_classes for b in datasets for c in optimizers}

    for dataset in datasets:
        for model in model_classes:
            for optimizer in optimizers:
                for seed in seeds:
                    loss, gradient, run_time, iteration = results[dataset, optimizer, model, seed]

                    losses[model, dataset, optimizer].append(loss)
                    gradients[model, dataset, optimizer].append(gradient)
                    run_times[model, dataset, optimizer].append(run_time)
                    iterations[model, dataset, optimizer].append(iteration)

    print('} & \\rotatebox{90}{'.join(optimizers) + '\\\\\n\\midrule')

    for model in model_classes:
        for dataset in datasets:

            print(f'{model.__name__} & {dataset.__name__} ', end='')
            for optimizer in optimizers:
                plt.figure(figsize=(4, 2.5))
                min_time = np.inf
                min_time_iter = np.inf
                min_time_loss = np.inf

                min_loss = np.inf
                min_loss_time = np.inf
                min_loss_iter = np.inf

                for seed in seeds:
                    if run_times[model, dataset, optimizer][seed] < min_time:
                        min_time = run_times[model, dataset, optimizer][seed]
                        min_time_iter = iterations[model, dataset, optimizer][seed]
                        min_time_loss = losses[model, dataset, optimizer][seed][-1]

                    if losses[model, dataset, optimizer][seed][-1] < min_loss:
                        min_loss = losses[model, dataset, optimizer][seed][-1]
                        min_loss_time = run_times[model, dataset, optimizer][seed]
                        min_loss_iter = iterations[model, dataset, optimizer][seed]

                    plt.plot(losses[model, dataset, optimizer][seed])
                plt.ylim(ymin=max(0, min(itertools.chain.from_iterable(losses[model, dataset, optimizer])) - 0.02),
                         ymax=min(5, max(itertools.chain.from_iterable(losses[model, dataset, optimizer])) + 0.02))
                plt.xlabel('Iterations')
                plt.ylabel('Loss (Mean NLL)')
                plt.title(f'{model.__name__} {dataset.__name__} {optimizer}')
                plt.savefig(f'plots/{model.__name__}_{dataset.__name__}_{optimizer}.pdf', bbox_inches='tight')
                plt.close()

                if min_time_iter < 500:
                    time = f'{float(f"{min_time:.2g}"):g}' if min_time < 10 else round(min_time)
                    print(f' & \\textbf{{{time}}}', end='')
                    # print(f' & \\textbf{{{min_time:.2f}}}', end='')
                    # print(f' & {min_time_iter}', end='')

                else:
                    # print(f' & {min_time_loss:.2f}', end='')
                    time = f'{float(f"{min_loss_time:.2g}"):g}' if min_loss_time < 10 else round(min_loss_time)
                    print(f' & {time}', end='')

            print('\\\\')


if __name__ == '__main__':
    plot('results/item_identity_results.pt')
    plot('results/item_feature_results.pt')
