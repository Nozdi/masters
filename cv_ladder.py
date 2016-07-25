#!/usr/bin/env python
import argparse

from collections import OrderedDict
from itertools import product
import json

import yaml
import pandas as pd
import numpy as np
np.random.seed(1)  # for reproducibility

from fuel.datasets import IndexableDataset
from joblib import (
    Parallel,
    delayed,
)
from sklearn.grid_search import ParameterGrid
from nested_kfold import nested_kfold
from metrics import (
    binarize_y,
    create_score_dict
)
from ladder.train import train_ladder


LADDERS_CONFIG = 'ladder_models.yaml'
DATASET = 'data/dataset.csv'
TARGET_NAME = 'MalignancyCharacter'
N_CORES = 4

with open(LADDERS_CONFIG, 'r') as fp:
    configs = yaml.load(fp)


df = pd.read_csv(DATASET)
y = df[TARGET_NAME].values.astype(np.int)
y_bin = binarize_y(y)
indexes = nested_kfold(y, method='stratified')


class OvaDataset(IndexableDataset):
    def __init__(self, X, y, **kwargs):
        indexables = OrderedDict(
            [('features', X),
             ('targets', y)]
        )
        super(OvaDataset, self).__init__(indexables, **kwargs)


def validate_ladder(config, df, y, train_indexes, val_indexes, sub_name, name):
    _config = config.copy()
    _y = y.copy()
    X = df[_config.pop('x_features')].values.astype(np.float)
    binary = _config.pop('binary')
    score = {
        'config': json.dumps(config),
        'error': None,
    }
    if binary:
        _y = binarize_y(y)

    try:
        res, inputs = train_ladder(
            config,
            dataset={
                'ovadataset': OvaDataset(X, _y),
                'train_indexes': train_indexes,
                'val_indexes': val_indexes,
            },
            save_to='ladder/{}/{}'.format(sub_name, name)
        )
    except Exception as e:
        res = np.zeros((len(y[val_indexes]), 3))
        score['error'] = str(e)

    score.update(create_score_dict(
        binarize_y(y[val_indexes]), binarize_y(res.argmax(axis=1)),
    ))
    pd.Series(score).to_csv("./results/ladder/{}/{}/score.csv".format(sub_name, name))
    return score


def cv_ladders(configs, indexes, name):
    test_scores = []
    for idx, fold in enumerate(indexes):
        scores = Parallel(n_jobs=N_CORES)(
            delayed(validate_ladder)(
                config, df, y,
                nested_fold['train'],
                nested_fold['val'],
                sub_name=name,
                name="ova_{}_{}_{}".format(name, idx, inner_idx),
            )
            for inner_idx, (config, nested_fold) in
            enumerate(product(configs[:1], fold['nested_indexes']))
        )
        df_scores = pd.DataFrame(scores)
        df_scores.to_csv(
            "./results/ladder/{}/fold_{}_{}_scores.csv".format(name, name, idx),
            index=False
        )
        sorted_configs = df_scores.groupby('config').mean().sort_values(
            ['cost_matrix', 'SEN'], ascending=[True, False]
        )
        _config = yaml.safe_load(sorted_configs.index[0])
        X = df[_config.pop('x_features')].values.astype(np.float)

        _y = y.copy()
        if _config.pop('binary'):
            _y = binarize_y(y)

        res, inputs = train_ladder(
            _config,
            dataset={
                'ovadataset': OvaDataset(X, _y),
                'train_indexes': fold['train'],
                'val_indexes': fold['test'],
            },
            save_to='ladder/{}/ova_{}_{}'.format(name, name, idx)
        )
        binarized_y_true = binarize_y(y[fold['test']])
        binarized_y_pred = binarize_y(res.argmax(axis=1))
        test_scores.append(
            create_score_dict(binarized_y_true, binarized_y_pred)
        )
    results = pd.DataFrame(test_scores)
    results.to_csv("./results/ladder/{}/{}_all.csv".format(name, name), index=False)
    return results


def cv_all_ladders(configs, indexes):
    for name, config_grid in configs.iteritems():
        cv_ladders(list(ParameterGrid(config_grid)), indexes, name=name)


def find_best_model(configs):
    results = {}
    for name in configs:
        filename = "./results/ladder/{}/{}_all.csv".format(name, name)
        results[name] = pd.read_csv(filename)

    best_config = sorted(
        results, key=lambda config: results[config]['cost_matrix'].mean())[0]
    return {
        'config_name': best_config,
        'df': results[best_config]
    }


def calculate_best_model(config_grid, indexes):
    configs = list(ParameterGrid(config_grid))
    scores = Parallel(n_jobs=N_CORES)(
        delayed(validate_ladder)(
            config, df, y,
            fold['train'],
            fold['test'],
            name="ova_final_{}".format(idx),
        )
        for idx, (config, fold) in
        enumerate(product(configs, indexes))
    )
    df_scores = pd.DataFrame(scores)
    df_scores.to_csv(
        "./results/ladder/best_scores.csv", index=False
    )
    sorted_configs = df_scores.groupby('config').mean().sort_values(
        ['cost_matrix', 'SEN'], ascending=[True, False]
    )
    return sorted_configs.iloc[0]


def get_best_model_config():
    df_scores = pd.read_csv("./results/ladder/best_scores.csv")
    sorted_configs = df_scores.groupby('config').mean().sort_values(
        ['cost_matrix', 'SEN'], ascending=[True, False]
    )
    return sorted_configs.iloc[0].name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to do cross validation using ladder network on OVA dataset')
    parser.add_argument(
        '--train', dest='train', action='store_true',
        help='if this flag is set script will do CV from scratch;\n'
             'CAVEAT: it might take up to 1 month using 20 cores')
    parser.add_argument(
        '--train-best', dest='train_best', action='store_true',
        help='if this flag is set script will do CV to find best model'
             ' using nested CV results\n'
             'CAVEAT: it might take up to 24 hours using 20 cores')
    parser.add_argument(
        '--get-cv-results', dest='cv_results', action='store_true',
        help='if this flag is set script will return best model'
             ' settings and CV results')

    args = parser.parse_args()
    if args.train:
        cv_all_ladders(configs, indexes)

    if args.train_best:
        config_grid = configs[find_best_model(configs)['config_name']]
        calculate_best_model(config_grid, indexes)

    if args.cv_results:
        score_df = find_best_model(configs)['df']
        print get_best_model_config()
        print score_df
        print pd.DataFrame(
            {'mean': score_df.mean(), 'std': score_df.std()}
        )
