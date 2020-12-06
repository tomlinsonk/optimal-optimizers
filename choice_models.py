import time
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from scipy.stats import stats
from tqdm import tqdm


EPOCHS = 500

class DataLoader:
    """
    Simplified, faster DataLoader.
    From https://github.com/arjunsesh/cdm-icml with minor tweaks.
    """
    def __init__(self, data, batch_size=None, shuffle=False):
        self.data = data
        self.data_size = data[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.counter = 0
        self.stop_iteration = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop_iteration:
            self.stop_iteration = False
            raise StopIteration()

        if self.batch_size is None or self.batch_size == self.data_size:
            self.stop_iteration = True
            return self.data
        else:
            i = self.counter
            bs = self.batch_size
            self.counter += 1
            batch = [item[i * bs:(i + 1) * bs] for item in self.data]
            if self.counter * bs >= self.data_size:
                self.counter = 0
                self.stop_iteration = True
                if self.shuffle:
                    random_idx = np.arange(self.data_size)
                    np.random.shuffle(random_idx)
                    self.data = [item[random_idx] for item in self.data]
            return batch


class ChoiceModel(nn.Module):
    def loss(self, y_hat, y):
        """
        The error in inferred log-probabilities given observations
        :param y_hat: log(choice probabilities)
        :param y: observed choices
        :return: the loss
        """
        return nnf.nll_loss(y_hat, y)

    def accuracy(self, y_hat, y):
        """
        Compute accuracy (fraction of choice set correctly predicted)
        :param y_hat: log(choice probabilities)
        :param y: observed choices
        :return: the accuracy
        """
        return (y_hat.argmax(1).int() == y.int()).float().mean()

    def mean_relative_rank(self, y_hat, y):
        """
        Compute mean rank of correct answer in output sorted by probability
        :param y_hat:
        :param y:
        :return:
        """
        return np.mean(self.relative_ranks(y_hat, y))

    def relative_ranks(self, y_hat, y):
        """
        Compute mean rank of correct answer in output sorted by probability
        :param y_hat:
        :param y:
        :return:
        """
        y_hat = y_hat.squeeze()
        y = y.squeeze()

        choice_set_lengths = np.array((~torch.isinf(y_hat)).sum(1))
        ranks = stats.rankdata(-y_hat.detach().numpy(), method='average', axis=1)[np.arange(len(y)), y] - 1

        return ranks / (choice_set_lengths - 1)

    @property
    def num_params(self):
        return sum(param.numel() for param in self.parameters() if param.requires_grad)


class ItemIdentityChoiceModel(ChoiceModel, ABC):
    @abstractmethod
    def forward(self, choice_sets):
        """
        Compute log(choice probabilities) of items in choice sets
        :param choice_sets: the choice sets
        :return: log(choice probabilities) over every choice set
        """
        pass


class ItemFeatureChoiceModel(ChoiceModel, ABC):
    @abstractmethod
    def forward(self, choice_set_features, choice_set_lengths):
        """
        Compute log(choice probabilities) of items in choice sets
        :param choice_set_features: the choice sets, with item features
        :param choice_set_lengths: number of items in each choice set
        :return: log(choice probabilities) over every choice set
        """

class Logit(ItemIdentityChoiceModel):
    name = 'logit'
    table_name = 'Logit'

    def __init__(self, num_items):
        """
        Initialize an MNL model for inference
        :param num_items: size of U
        """
        super().__init__()
        self.num_items = num_items

        self.utilities = nn.Parameter(torch.zeros(self.num_items, 1))

    def forward(self, choice_sets):
        """
        Compute log(choice probabilities) of items in choice sets
        :param choice_sets: the choice sets
        :return: log(choice probabilities) over every choice set
        """

        utilities = self.utilities * choice_sets
        utilities[choice_sets == 0] = -np.inf

        return nnf.log_softmax(utilities, 1)


class CDM(ItemIdentityChoiceModel):
    name = 'cdm'
    table_name = 'CDM'

    def __init__(self, num_items):
        super().__init__()
        self.num_items = num_items
        self.pulls = nn.Parameter(torch.zeros(self.num_items, self.num_items))

    def forward(self, choice_sets):
        """
        Compute log(choice probabilities) of items in choice sets
        :param choice_sets: the choice sets
        :return: log(choice probabilities) over every choice set
        """
        utilities = ((choice_sets.squeeze() @ self.pulls).unsqueeze(-1) - torch.diag(self.pulls)[:, None]) * choice_sets
        utilities[choice_sets == 0] = -np.inf

        return nnf.log_softmax(utilities, 1)


class ConditionalLogit(ItemFeatureChoiceModel):
    name = 'conditional-logit'
    table_name = 'CL'

    def __init__(self, num_item_feats):
        """
        :param num_item_feats: number of item features
        """
        super().__init__()
        self.num_item_feats = num_item_feats
        self.theta = nn.Parameter(torch.zeros(self.num_item_feats))

    def forward(self, choice_set_features, choice_set_lengths):
        """
        Compute log(choice probabilities) of items in choice sets
        :param choice_set_features: the choice sets, with item features
        :param choice_set_lengths: number of items in each cchoice set
        :return: log(choice probabilities) over every choice set
        """
        batch_size, max_choice_set_len, num_feats = choice_set_features.size()

        utilities = (self.theta * choice_set_features).sum(-1)
        utilities[torch.arange(max_choice_set_len)[None, :] >= choice_set_lengths[:, None]] = -np.inf

        return nnf.log_softmax(utilities, 1)


class LCL(ItemFeatureChoiceModel):
    name = 'lcl'
    table_name = 'LCL'

    def __init__(self, num_item_feats):
        """
        :param num_item_feats: number of item features
        """
        super().__init__()
        self.num_item_feats = num_item_feats
        self.theta = nn.Parameter(torch.zeros(self.num_item_feats))
        self.A = nn.Parameter(torch.zeros(self.num_item_feats, self.num_item_feats))

    def forward(self, choice_set_features, choice_set_lengths):
        batch_size, max_choice_set_len, num_feats = choice_set_features.size()

        # Compute mean of each feature over each choice set
        mean_choice_set_features = choice_set_features.sum(1) / choice_set_lengths.unsqueeze(-1)

        # Compute context effect in each sample
        context_effects = self.A @ mean_choice_set_features.unsqueeze(-1)

        # Compute context-adjusted utility of every item
        utilities = ((self.theta.unsqueeze(-1) + context_effects).view(batch_size, 1, -1) * choice_set_features).sum(-1)
        utilities[torch.arange(max_choice_set_len).unsqueeze(0) >= choice_set_lengths.unsqueeze(-1)] = -np.inf

        return nn.functional.log_softmax(utilities, 1)


def fit(model, data, optimizer, show_progress=True):
    """
    Fit a choice model to data using the given optimizer.

    :param model: a nn.Module
    :param data:
    :param epochs: number of optimization epochs
    :param learning_rate: step size hyperparameter for Adam
    :param weight_decay: regularization hyperparameter for Adam
    :param show_live_loss: if True, add loss/accuracy to progressbar. Adds ~50% overhead
    """
    torch.set_num_threads(1)

    choices = data[-1]

    progress_bar = tqdm(range(EPOCHS), total=EPOCHS) if show_progress else range(EPOCHS)

    start = time.time()
    iterations = 0
    losses = []

    for epoch in progress_bar:
        model.train()

        optimizer.zero_grad()

        loss = model.loss(model(*data[:-1]), choices)
        losses.append(loss.detach().item())

        loss.backward(retain_graph=None if epoch != EPOCHS - 1 else True)

        with torch.no_grad():
            gradient = torch.stack([(item.grad ** 2).sum() for item in model.parameters()]).sum()

            if gradient.item() < 10 ** -8:
                break

        optimizer.step()

        if show_progress:
            progress_bar.set_description(f'Loss: {loss.item():.4f}, Grad: {gradient.item():.3e}. Epochs')

        iterations += 1

    run_time = time.time() - start

    loss = model.loss(model(*data[:-1]), choices)
    losses.append(loss.detach().item())

    loss.backward()
    with torch.no_grad():
        gradient = torch.stack([(item.grad ** 2).sum() for item in model.parameters()]).sum()

    if show_progress:
        print('Done. Final gradient:', gradient.item(), 'Final NLL:', loss.item() * len(data[-1]))

    return losses, gradient.item(), run_time, iterations
