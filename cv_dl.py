#!/usr/bin/env python
import argparse

import json
import os
from itertools import product

import yaml
import numpy as np
import pandas as pd

seed = 1
np.random.seed(seed)  # for reproducibility
from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(seed=seed)
import theano.tensor as T


from metrics import (
    binarize_y,
    create_score_dict,
)
from nested_kfold import nested_kfold

from joblib import (
    Parallel,
    delayed,
)
from keras.models import Sequential
from keras.layers.core import (
    Dense,
    Dropout
)
from keras.utils.np_utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from sklearn.grid_search import ParameterGrid


ANN_GRID = {
    'batch_size': [25, 50],
    'nb_epoch': range(200, 501, 25),
    'optimizer': ['sgd', 'adam'],
    'dropout_1': [0.1, 0.15, 0.2],
    'dropout_2': [0.45, 0.5, 0.55],
    'output_dim': [2]
}
N_CORES = 20
DATASET_FILEPATH = 'data/dataset.csv'
RESULTS_DIRECTORY = './results/dl'

X_FEATURES = ['Color', 'Ca125', 'AgeAfterMenopause']
TARGET = 'MalignancyCharacter'

df = pd.read_csv(DATASET_FILEPATH)
y = df[TARGET].values.astype(np.int)
y_bin = binarize_y(y)
indexes = nested_kfold(y, method='stratified')


def create_dir():
    if not os.path.isdir(RESULTS_DIRECTORY):
        os.makedirs(RESULTS_DIRECTORY)


def weighted_cost(y_true, y_pred):
    benign_mask = T.eq(y_true[:, 0], 1).nonzero()[0]
    malignant_mask = T.neq(y_true[:, 0], 1).nonzero()[0]
    return T.concatenate(
        [T.nnet.categorical_crossentropy(
            y_pred[benign_mask], y_true[benign_mask]),
         2 * T.nnet.categorical_crossentropy(
            y_pred[malignant_mask], y_true[malignant_mask])]
    )


def weighted_cost_binary(y_true, y_pred):
    multiplier = y_true[:, 0] + y_true[:, 1] * 2
    return T.mean(
        multiplier * T.nnet.binary_crossentropy(y_pred[:, 1], y_true[:, 1])
    )


def prepare_ann(
    optimizer, input_dim, output_dim=3,
    dropout_1=0.2, dropout_2=0.5
):
    np.random.seed(seed)  # for reproducibility
    model = Sequential()
    model.add(BatchNormalization(input_shape=(input_dim, )))
    model.add(Dropout(dropout_1))
    model.add(Dense(3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_2))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss=weighted_cost, optimizer=optimizer)
    return model


def pred_ann(
    df, train_idx, val_idx, config,
    feature_list=X_FEATURES, target=TARGET
):
    output_dim = config['output_dim']
    model = prepare_ann(
        config['optimizer'],
        input_dim=len(feature_list), output_dim=output_dim,
        dropout_1=config['dropout_1'], dropout_2=config['dropout_2'])

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    X_train = train_df[feature_list].values
    if output_dim == 3:
        y_train = to_categorical(train_df[target].values)
    else:
        y_train = to_categorical((train_df[target].values > 0).astype(int))

    model.fit(
        X_train,
        y_train,
        batch_size=config['batch_size'], nb_epoch=config['nb_epoch'],
        verbose=0
    )

    X_val = val_df[feature_list].values
    return model.predict_proba(X_val, verbose=0).argmax(axis=1)


def validate_ann(
    config, df, y, train_indexes, val_indexes, pred_function, seed=1
):
    np.random.seed(seed)  # for reproducibility
    score = {
        'config': json.dumps(config),
    }
    result = pred_function(df, train_indexes, val_indexes, config)
    score.update(create_score_dict(
        binarize_y(y[val_indexes]), binarize_y(result),
    ))
    return score


def cv_nn(indexes, grid, pred_function=pred_ann, df=df, y=y):
    test_scores = []
    for idx, fold in enumerate(indexes):
        configs = list(ParameterGrid(grid))
        nested_cv_results = Parallel(n_jobs=N_CORES)(
            delayed(validate_ann)(
                config, df, y,
                nested_fold['train'],
                nested_fold['val'],
                pred_function=pred_function,
                seed=seed,
            )
            for config, nested_fold in
            product(configs, fold['nested_indexes'])
        )
        df_scores = pd.DataFrame(nested_cv_results)
        df_scores.to_csv(
            '{}/fold_dl_{}_scores.csv'.format(RESULTS_DIRECTORY, idx), index=False
        )
        sorted_configs = df_scores.groupby('config').mean().sort_values(
            ['cost_matrix', 'SEN', 'NPV'], ascending=[True, False, False]
        )
        config = yaml.safe_load(sorted_configs.index[0])
        np.random.seed(seed)
        pred = pred_function(
            df, fold['train'], fold['test'], config
        )
        test_scores.append(
            create_score_dict(y_bin[fold['test']], binarize_y(pred))
        )
        print test_scores[-1]
    results = pd.DataFrame(test_scores)
    results.to_csv('{}/dl_all.csv'.format(RESULTS_DIRECTORY), index=False)
    return results


def generate_best(indexes, grid, pred_function=pred_ann, df=df, y=y):
    configs = list(ParameterGrid(grid))
    scores = Parallel(n_jobs=N_CORES)(
        delayed(validate_ann)(
            config, df, y,
            fold['train'],
            fold['test'],
            pred_function=pred_ann,
        )
        for idx, (config, fold) in
        enumerate(product(configs, indexes))
    )
    df_scores = pd.DataFrame(scores)
    df_scores.to_csv(
        '{}/best_scores.csv'.format(RESULTS_DIRECTORY), index=False
    )
    sorted_configs = df_scores.groupby('config').mean().sort_values(
        ['cost_matrix', 'SEN', 'NPV'], ascending=[True, False, False]
    )

    best_config = yaml.safe_load(sorted_configs.index[0])

    np.random.seed(seed)
    model = prepare_ann(
        best_config['optimizer'], input_dim=len(X_FEATURES),
        output_dim=best_config['output_dim'],
        dropout_1=best_config['dropout_1'],
        dropout_2=best_config['dropout_2'])
    X = df[X_FEATURES].values
    y = to_categorical((df[TARGET].values > 0).astype(int))

    model.fit(
        X,
        y,
        batch_size=best_config['batch_size'], nb_epoch=best_config['nb_epoch'],
        verbose=0
    )
    with open('{}/model_config.yaml'.format(RESULTS_DIRECTORY), 'w') as fp:
        fp.write(model.to_yaml())

    model.save_weights('{}/model_weights.h5'.format(RESULTS_DIRECTORY))
    return best_config


def get_best_config():
    df_scores = pd.read_csv(
        '{}/best_scores.csv'.format(RESULTS_DIRECTORY)
    )
    sorted_configs = df_scores.groupby('config').mean().sort_values(
        ['cost_matrix', 'SEN', 'NPV'], ascending=[True, False, False]
    )
    return sorted_configs.iloc[0].name


def best_model():
    return pd.read_csv('{}/dl_all.csv'.format(RESULTS_DIRECTORY))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to do cross validation using deep learning on OVA dataset')
    parser.add_argument(
        '--train', dest='train', action='store_true',
        help='if this flag is set script will do CV from scratch;\n'
             'CAVEAT: it might take up to 2-3 h using 20 cores')
    parser.add_argument(
        '--train-best', dest='train_best', action='store_true',
        help='if this flag is set script will do CV to find best model'
             ' using nested CV results\n'
             'CAVEAT: it might take up to 1 hour using 20 cores')
    parser.add_argument(
        '--get-cv-results', dest='cv_results', action='store_true',
        help='if this flag is set script will return best model'
             ' settings and CV results')
    args = parser.parse_args()

    create_dir()
    if args.train:
        cv_nn(indexes, ANN_GRID)
    if args.train_best:
        generate_best(indexes, ANN_GRID)
    if args.cv_results:
        print get_best_config()
        df = best_model()
        print df
        print pd.DataFrame(
            {'mean': df.mean(), 'std': df.std()}
        )
