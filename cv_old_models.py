#!/usr/bin/env python
import argparse

import pandas as pd
import numpy as np
np.random.seed(1)  # for reproducibility

from nested_kfold import nested_kfold
from metrics import (
    create_score_dict,
    binarize_y,
)


DATASET = 'data/dataset.csv'
TARGET_NAME = 'MalignancyCharacter'


df = pd.read_csv(DATASET)
y = df[TARGET_NAME].values.astype(np.int)
y_bin = binarize_y(y)
indexes = nested_kfold(y, method='stratified')


def cv_old_models(df, indexes):
    models = [
        'TimmermannBin',
        'LR1Bin',
        'LR2Bin',
        'SMBin',
        'AdnexBin',
    ]
    models_dict = dict.fromkeys(models)
    for model in models:
        test_scores = []
        for fold in indexes:
            y_true = y_bin[fold['test']]
            y_pred = df[model].values.astype(np.int)[fold['test']]
            test_scores.append(create_score_dict(y_true, y_pred))
        models_dict[model] = pd.DataFrame(test_scores)
    return models_dict


def create_std_mean_score(df):
    return pd.DataFrame(
        {'mean': df.mean(), 'std': df.std()}
    )


def print_result(old_models_cv_results, name):
    score_df = old_models_cv_results[name]
    print score_df
    print create_std_mean_score(score_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to do cross validation using old models on OVA dataset')
    parser.add_argument(
        '--tim', dest='tim', action='store_true',
        help='Getting CV results using Timmerman logistic regression model')
    parser.add_argument(
        '--lr1', dest='lr1', action='store_true',
        help='Getting CV results using LR1 IOTA model')
    parser.add_argument(
        '--lr2', dest='lr2', action='store_true',
        help='Getting CV results using LR2 IOTA model')
    parser.add_argument(
        '--sm', dest='sm', action='store_true',
        help='Getting CV results using SM model')
    parser.add_argument(
        '--adnex', dest='adnex', action='store_true',
        help='Getting CV results using Andex model')

    parser.add_argument(
        '--all', dest='all', action='store_true',
        help='Getting CV results using Timmerman / LR1 / LR2 / SM / Adnex model')

    old_models_cv_results = cv_old_models(df, indexes)
    args = parser.parse_args()
    if args.tim or args.all:
        print "Timmerman Model:"
        print_result(old_models_cv_results, 'TimmermannBin')
    if args.lr1 or args.all:
        print "LR1 Model:"
        print_result(old_models_cv_results, 'LR1Bin')
    if args.lr2 or args.all:
        print "LR2 Model:"
        print_result(old_models_cv_results, 'LR2Bin')
    if args.sm or args.all:
        print "SM Model:"
        print_result(old_models_cv_results, 'SMBin')
    if args.adnex or args.all:
        print "Adnex Model:"
        print_result(old_models_cv_results, 'AdnexBin')
