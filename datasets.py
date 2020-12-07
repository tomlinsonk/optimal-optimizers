import os
import pickle
from abc import ABC, abstractmethod

import scipy.io as sio
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from itertools import chain, combinations

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

DATA_DIR = config['datadir']


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def all_choice_set_indices(n_items):
    indices = list(range(n_items))
    return chain.from_iterable(combinations(indices, size) for size in range(2, n_items+1))


class Dataset(ABC):
    item_names = []

    @classmethod
    @abstractmethod
    def load(cls):
        ...

    @classmethod
    def one_hot_encode(cls, df, col_name):
        """
        One-hot encode a categorical feature in a pandas dataframe in-place
        :param df: the pandas DataFrame
        :param col_name: the name of the column
        """
        df = pd.concat([df, pd.get_dummies(df[col_name], prefix=col_name, drop_first=True)], axis=1)
        df.drop([col_name], axis=1, inplace=True)
        return df


class ItemIdentityDataset(Dataset, ABC):
    @classmethod
    def load_pytorch(cls):
        choice_sets, choices, person_df = cls.load()
        choice_sets = torch.from_numpy(choice_sets)[:, :, None].long()
        choices = torch.tensor(choices).long()

        if len(person_df.index) > 0:
            person_df = pd.DataFrame(preprocessing.StandardScaler().fit_transform(person_df), columns=person_df.columns, index=person_df.index)

        return choice_sets, choices, person_df


class ItemFeatureDataset(Dataset, ABC):
    @classmethod
    def load_pytorch(cls):
        choice_set_features, choice_set_lengths, choices, person_df = cls.load()
        choice_set_features = torch.from_numpy(choice_set_features).float()
        choice_set_lengths = torch.from_numpy(choice_set_lengths).long()
        choices = torch.tensor(choices).long()

        all_feature_vecs = choice_set_features[torch.arange(choice_set_features.size(1))[None, :] < choice_set_lengths[:, None]]
        means = all_feature_vecs.mean(0)
        stds = all_feature_vecs.std(0)

        choice_set_features[torch.arange(choice_set_features.size(1))[None, :] < choice_set_lengths[:, None]] -= means
        choice_set_features[torch.arange(choice_set_features.size(1))[None, :] < choice_set_lengths[:, None]] /= stds

        person_df = pd.DataFrame(preprocessing.StandardScaler().fit_transform(person_df), columns=person_df.columns, index=person_df.index)

        return choice_set_features, choice_set_lengths, choices, person_df


class SFWork(ItemIdentityDataset):
    name = 'sf-work'

    item_names = ['Drive Alone', 'Shared Ride 2', 'Shared Ride 3+', 'Transit', 'Bike', 'Walk']

    @classmethod
    def load(cls):
        user_feature_names = ['femdum', 'age', 'corredis', 'dist', 'drlicdum', 'famtype', 'hhinc', 'hhowndum', 'hhsize',
                              'nm12to16', 'nm5to11', 'nmlt5', 'noncadum', 'numadlt', 'numemphh', 'numveh', 'rsempden',
                              'rspopden', 'vehavdum', 'vehbywrk', 'wkccbd', 'wkempden', 'wknccbd', 'wkpopden']

        sf_work = sio.loadmat(f'{DATA_DIR}/SF/SF-raw/SF_HBW/SFMTCWork6.mat')

        indivs = np.unique(np.concatenate((sf_work['hhid'], sf_work['perid']), axis=1), axis=0)
        indiv_indices = {(hhid, perid): [] for hhid, perid in indivs}

        for i in range(len(sf_work['hhid'])):
            indiv_indices[sf_work['hhid'][i][0], sf_work['perid'][i][0]].append(i)

        choice_sets = np.zeros((len(indivs), 6), dtype=int)
        for i, (hhid, perid) in enumerate(indivs):
            for j in indiv_indices[hhid, perid]:
                choice_sets[i, sf_work['alt'][j] - 1] = 1

        choices = np.zeros((len(indivs), 1), dtype=int)
        for i, (hhid, perid) in enumerate(indivs):
            for j in indiv_indices[hhid, perid]:
                if sf_work['chosen'][j]:
                    choices[i] = sf_work['alt'][j] - 1

        person_features = np.zeros((len(choice_sets), len(user_feature_names)), dtype=float)
        for i, (hhid, perid) in enumerate(indivs):
            j = indiv_indices[hhid, perid][0]
            for feat_idx, feat in enumerate(user_feature_names):
                person_features[i, feat_idx] = sf_work[feat][j]

        multi_idx = pd.MultiIndex.from_tuples([tuple(row) for row in indivs], names=('hhid', 'perid'))

        person_df = cls.one_hot_encode(pd.DataFrame(person_features, index=multi_idx, columns=user_feature_names), 'famtype')

        return choice_sets, choices, person_df


class SFShop(ItemIdentityDataset):
    name = 'sf-shop'

    item_names = ['Transit', 'SR2', 'SR3+',
                  'Drive Alone and SR', 'SR2 and SR3+', 'Bike', 'Walk', 'Drive Alone']

    @classmethod
    def load(cls):
        user_feature_names = ['DISTANCE', 'D_DENS', 'HHSIZE', 'HHSIZE5', 'INCOME', 'O_DENS', 'URBAN', 'VEHICLES']

        sf_shop = sio.loadmat(f'{DATA_DIR}/SF/SF-raw/SF_HBShO/SFHBSHOw5.mat')

        indivs = np.unique(sf_shop['ID'])
        indiv_indices = {id: [] for id in indivs}

        for i in range(len(sf_shop['ID'])):
            indiv_indices[sf_shop['ID'][i][0]].append(i)

        choice_sets = np.zeros((len(indivs), len(cls.item_names)), dtype=int)
        for i, id in enumerate(indivs):
            for j in indiv_indices[id]:
                choice_sets[i, sf_shop['ALTID'][j] - 1] = 1

        choices = np.zeros((len(indivs), 1), dtype=int)
        for i, id in enumerate(indivs):
            for j in indiv_indices[id]:
                if sf_shop['CHOSEN'][j]:
                    choices[i] = sf_shop['ALTID'][j] - 1

        person_features = np.zeros((len(choice_sets), len(user_feature_names)), dtype=float)
        for i, id in enumerate(indivs):
            j = indiv_indices[id][0]
            for feat_idx, feat in enumerate(user_feature_names):
                person_features[i, feat_idx] = sf_shop[feat][j]

        person_df = pd.DataFrame(person_features, index=indivs, columns=user_feature_names)

        return choice_sets, choices, person_df


class Sushi(ItemIdentityDataset):
    name = 'sushi'
    item_names = ['ebi', 'anago', 'maguro', 'ika', 'uni', 'tako', 'ikura', 'tamago', 'toro', 'amaebi', 'hotategai', 'tai',
             'akagai', 'hamachi', 'awabi', 'samon', 'kazunoko', 'shako', 'saba', 'chu_toro', 'hirame', 'aji', 'kani',
             'kohada', 'torigai', 'unagi', 'tekka_maki', 'kanpachi', 'mirugai', 'kappa_maki', 'geso', 'katsuo',
             'iwashi', 'hokkigai', 'shimaaji', 'kanimiso', 'engawa', 'negi_toro', 'nattou_maki', 'sayori',
             'takuwan_maki', 'botanebi', 'tobiko', 'inari', 'mentaiko', 'sarada', 'suzuki', 'tarabagani',
             'ume_shiso_maki', 'komochi_konbu', 'tarako', 'sazae', 'aoyagi', 'toro_samon', 'sanma', 'hamo', 'nasu',
             'shirauo', 'nattou', 'ankimo', 'kanpyo_maki', 'negi_toro_maki', 'gyusashi', 'hamaguri', 'basashi', 'fugu',
             'tsubugai', 'ana_kyu_maki', 'hiragai', 'okura', 'ume_maki', 'sarada_maki', 'mentaiko_maki', 'buri',
             'shiso_maki', 'ika_nattou', 'zuke', 'himo', 'kaiware', 'kurumaebi', 'mekabu', 'kue', 'sawara', 'sasami',
             'kujira', 'kamo', 'himo_kyu_maki', 'tobiuo', 'ishigakidai', 'mamakari', 'hoya', 'battera', 'kyabia',
             'karasumi', 'uni_kurage', 'karei', 'hiramasa', 'namako', 'shishamo', 'kaki']

    @classmethod
    def load(cls):
        user_feature_names = ['gender', 'age', 'survey_time', 'child_prefecture', 'child_region',
                              'child_east/west', 'prefecture', 'region', 'east/west', 'same_prefecture']

        rankings = np.loadtxt(f'{DATA_DIR}/sushi3-2016/sushi3b.5000.10.order', skiprows=1, usecols=range(2, 12),
                              dtype=int)

        person_features = np.loadtxt(f'{DATA_DIR}/sushi3-2016/sushi3.udata')

        person_df = pd.DataFrame(person_features[:, 1:], index=person_features[:, 0], columns=user_feature_names)

        categorical_feats = ['age', 'child_prefecture', 'child_region', 'prefecture', 'region']
        for feat in categorical_feats:
            person_df = cls.one_hot_encode(person_df, feat)

        choice_sets = np.zeros((len(rankings), 100), dtype=int)
        choice_sets[np.arange(len(rankings))[:, None], rankings] = 1
        choices = rankings[:, 0][:, None]

        return choice_sets, choices, person_df


class FeatureSushi(ItemFeatureDataset):
    name = 'feature-sushi'

    @classmethod
    def load(cls):
        old_choice_sets, old_choices, person_df = Sushi.load()

        item_feats = np.loadtxt(f'{DATA_DIR}/sushi3-2016/sushi3.idata', usecols=[2, 3, 5, 6, 7, 8])
        item_df = pd.DataFrame(item_feats, columns=['style', 'major_group', 'oiliness', 'popularity', 'price', 'availability'])

        range_100 = np.arange(100)
        choice_set_indices = np.array([range_100[row == 1] for row in old_choice_sets])
        choice_set_features = np.array([item_feats[row] for row in choice_set_indices])

        choice_set_lengths = np.full(len(choice_set_features), 10)

        choices = np.array([np.searchsorted(choice_set_indices[i], old_choices[i])[0] for i in range(len(choice_set_features))])

        return choice_set_features, choice_set_lengths, choices, person_df


class Expedia(ItemFeatureDataset):
    name = 'expedia'

    @classmethod
    def load(cls):
        pickle_fname = f'{DATA_DIR}/pickles/expedia.pickle'
        if os.path.isfile(pickle_fname):
            with open(pickle_fname, 'rb') as f:
                return pickle.load(f)

        item_feats = ['prop_starrating', 'prop_review_score', 'prop_location_score1', 'price_usd', 'promotion_flag']
        chooser_feat_names = ['visitor_hist_starrating', 'visitor_hist_adr_usd', 'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool']

        df = pd.read_csv(f'{DATA_DIR}/expedia-personalized-sort/train.csv', usecols=['srch_id', 'prop_id', 'booking_bool'] + item_feats + chooser_feat_names)

        # Select only searches that result in a booking
        df = df[df.groupby(['srch_id'])['booking_bool'].transform(max) == 1]

        max_choice_set_size = df['srch_id'].value_counts().max()
        samples = df['srch_id'].nunique()
        n_feats = len(item_feats)

        choice_set_features = np.zeros((samples, max_choice_set_size, n_feats))
        choice_set_lengths = np.zeros(samples)
        choices = np.zeros(samples)

        chooser_feats = np.zeros((samples, len(chooser_feat_names)+1))

        for i, (srch_id, group) in tqdm(enumerate(df.groupby('srch_id')), total=samples):
            choice_set_length = len(group.index)
            choice_set_lengths[i] = choice_set_length

            item_features = group[item_feats].values
            item_features[np.isnan(item_features)] = 0

            choice_set_features[i, :choice_set_length] = item_features

            choices[i] = np.where(group['booking_bool'] == 1)[0]
            chooser_feats[i, :-1] = group[chooser_feat_names].values[0]

            # add 'has_prev_purchase' feature
            chooser_feats[i, -1] = int(not np.isnan(chooser_feats[i, 0]))

        # replace nan features with mean of non-nan
        for col in range(len(chooser_feat_names)):
            nan_idx = np.isnan(chooser_feats[:, col])
            mean = np.mean(chooser_feats[~nan_idx, col])
            chooser_feats[nan_idx, col] = mean

        person_df = pd.DataFrame(chooser_feats, columns=chooser_feat_names + ['has_prev_purchase'])

        with open(pickle_fname, 'wb') as f:
            pickle.dump((choice_set_features, choice_set_lengths, choices, person_df), f)

        return choice_set_features, choice_set_lengths, choices, person_df


class Allstate(ItemFeatureDataset):
    name = 'allstate'

    @classmethod
    def load(cls):
        pickle_fname = f'{DATA_DIR}/pickles/allstate.pickle'
        if os.path.isfile(pickle_fname):
            with open(pickle_fname, 'rb') as f:
                return pickle.load(f)

        df = pd.read_csv(f'{DATA_DIR}/allstate-purchase-prediction-challenge/train.csv')

        for feat in ['A', 'C', 'D', 'F', 'G']:
            df = cls.one_hot_encode(df, feat)

        item_feats = ['cost'] + [col for col in df.columns if col[0] in ['A', 'B', 'C', 'D', 'E', 'F', 'G'] and col != 'C_previous']
        print(item_feats)

        chooser_feat_names = ['group_size', 'homeowner', 'car_age', 'car_value', 'risk_factor', 'age_oldest', 'age_youngest', 'married_couple', 'C_previous', 'duration_previous']

        df['duration_previous'] = df['duration_previous'].fillna(0)

        max_choice_set_size = df['customer_ID'].value_counts().max()
        samples = df['customer_ID'].nunique()
        n_feats = len(item_feats)

        choice_set_features = np.zeros((samples, max_choice_set_size, n_feats))
        choice_set_lengths = np.zeros(samples)
        choices = np.zeros(samples)

        chooser_feats = []

        for i, (srch_id, group) in tqdm(enumerate(df.groupby('customer_ID')), total=samples):
            choice_set_length = len(group.index)
            choice_set_lengths[i] = choice_set_length

            item_features = group[item_feats].values
            item_features[np.isnan(item_features)] = 0

            choice_set_features[i, :choice_set_length] = item_features

            choices[i] = np.where(group['record_type'] == 1)[0]
            chooser_feats.append(group[chooser_feat_names].values[0])

        person_df = pd.DataFrame(chooser_feats, columns=chooser_feat_names)

        for feat in ['car_value', 'risk_factor', 'C_previous']:
            person_df = cls.one_hot_encode(person_df, feat)

        with open(pickle_fname, 'wb') as f:
            pickle.dump((choice_set_features, choice_set_lengths, choices, person_df), f)

        return choice_set_features, choice_set_lengths, choices, person_df



if __name__ == '__main__':
    Expedia.load()
    Allstate.load()