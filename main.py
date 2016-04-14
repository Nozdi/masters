from collections import OrderedDict
from operator import itemgetter
from itertools import (
    chain,
    product
)
import json

import yaml
import pandas as pd
import numpy as np

from tqdm import tqdm
from fuel.datasets import IndexableDataset
from joblib import (
    Parallel,
    delayed,
)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.grid_search import ParameterGrid
from sklearn.base import clone
from sklearn.metrics import confusion_matrix

from nested_kfold import nested_kfold
from metrics import (
    matrix_cost,
    binarize_y,
    PPV,
    NPV,
    SPC,
    SEN,
    ACC,
)
from ladder.run import train_own_dataset


LADDERS_CONFIG = 'ladder_models.yaml'
DATASET = 'data/dataset.csv'
TARGET_NAME = 'MalignancyCharacter'
N_CORES = 2


with open(LADDERS_CONFIG, 'r') as fp:
    configs = yaml.load(fp)


df = pd.read_csv(DATASET)
y = df[TARGET_NAME].values.astype(np.int)
indexes = nested_kfold(y, method='stratified')
first_fold = indexes[8]
first_nested_fold = first_fold['nested_indexes'][0]


class OvaDataset(IndexableDataset):
    def __init__(self, X, y, **kwargs):
        indexables = OrderedDict(
            [('features', X),
             ('targets', y)]
        )
        super(OvaDataset, self).__init__(indexables, **kwargs)


def validate_ladder(config, df, y, train_indexes, val_indexes):
    _config = config.copy()
    X = df[_config.pop('x_features')].values.astype(np.float)
    res, inputs = train_own_dataset(
        config,
        dataset={
            'ovadataset': OvaDataset(X, y),
            'train_indexes': train_indexes,
            'val_indexes': val_indexes,
        }
    )
    return {
        'score': matrix_cost(
            binarize_y(y[val_indexes]),
            binarize_y(res.argmax(axis=1)),
        ),
        'config': json.dumps(config),
    }


def create_score_dict(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred).astype(np.float)
    return {
        'PPV': PPV(cm),
        'NPV': NPV(cm),
        'SPC': SPC(cm),
        'SEN': SEN(cm),
        'ACC': ACC(cm),
        'cost_matrix': matrix_cost(y_true, y_pred)
    }


def cv_ladders(configs, indexes):
    test_scores = []
    all_configs = list(chain(
        *[list(ParameterGrid(grid)) for grid in configs.values()]
    ))

    for fold in tqdm(indexes):
        scores = Parallel(n_jobs=N_CORES)(
            delayed(validate_ladder)(
                config, df, y,
                nested_fold['train'],
                nested_fold['val']
            )
            for config, nested_fold in product(all_configs, fold['nested_indexes'])
        )
        df_scores = pd.DataFrame(scores)
        sorted_configs = df_scores.groupby('config').mean().sort_values('score')
        _config = yaml.safe_load(sorted_configs.index[0])
        X = df[_config.pop('x_features')].values.astype(np.float)
        res, inputs = train_own_dataset(
            _config,
            dataset={
                'ovadataset': OvaDataset(X, y),
                'train_indexes': fold['train'],
                'val_indexes': fold['test'],
            }
        )
        binarized_y_true = binarize_y(y[fold['test']])
        binarized_y_pred = binarize_y(res.argmax(axis=1))
        test_scores.append(
            create_score_dict(binarized_y_true, binarized_y_pred)
        )
    return pd.DataFrame(test_scores)


def cv_old_models(df, indexes):
    models = [
        'TimmermannBin',
        'LR1Bin',
        'LR2Bin',
        'SMBin',
        'GiradsDiagBin',
        'AdnexBin',
    ]
    models_dict = dict.fromkeys(models)
    for model in models:
        test_scores = []
        for fold in indexes:
            y_true = binarize_y(y[fold['test']])
            y_pred = df[model].values.astype(np.int)[fold['test']]
            test_scores.append(create_score_dict(y_true, y_pred))

        score_df = pd.DataFrame(test_scores)
        models_dict[model] = pd.DataFrame(
            {'mean': score_df.mean(), 'std': score_df.std()}
        )
    return models_dict


def cv_nn(indexes):
    pass


clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=1)
)
# df.loc[:, TARGET_NAME][df[TARGET_NAME] == 2] = 1
# y = df[TARGET_NAME].values.astype(np.int)


grid = {
    'logisticregression__C': [0.1, 1, 10, 100],
    'logisticregression__penalty': ['l1', 'l2'],
    'features': [['Pap', 'Ca125', 'ADimension', 'Color', 'Menopause']]

}


def validate_one_sk(df, y, base_estimator, config, train_indexes, val_indexes):
    _config = config.copy()
    X = df[_config.pop('features')].values.astype(np.float)
    clf = clone(base_estimator).set_params(**_config)
    clf.fit(X[train_indexes], y[train_indexes])
    return matrix_cost(
        y[val_indexes],
        clf.predict(X[val_indexes])
    )


def cv_sk(indexes, base_estimator, grid):
    test_scores = []
    for fold in tqdm(indexes):
        nested_cv_results = []
        for config in ParameterGrid(grid):
            # scores = Parallel(n_jobs=2)(
            #     delayed(validate_one_sk)(
            #         df, y, base_estimator, config,
            #         nested_fold['train'],
            #         nested_fold['val']
            #     )
            #     for nested_fold in fold['nested_indexes']
            # )
            scores = []
            for nested_fold in fold['nested_indexes']:
                _config = config.copy()
                X = df[_config.pop('features')].values.astype(np.float)
                clf = clone(base_estimator).set_params(**_config)
                clf.fit(X[nested_fold['train']], y[nested_fold['train']])
                score = matrix_cost(
                    binarize_y(y[nested_fold['val']]),
                    binarize_y(clf.predict(X[nested_fold['val']])),
                )
                scores.append(score)
            nested_cv_results.append({
                'config': config,
                'score': np.mean(scores)
            })

        best = sorted(nested_cv_results, key=itemgetter('score'))[0]
        _config = best['config']
        X = df[_config.pop('features')].values.astype(np.float)
        clf = clone(base_estimator).set_params(**_config)
        clf.fit(X[fold['train']], y[fold['train']])

        binarized_y_true = binarize_y(y[fold['test']])
        binarized_y_pred = binarize_y(clf.predict(X[fold['test']]))
        cm = confusion_matrix(binarized_y_true, binarized_y_pred).astype(np.float)
        score = {
            'PPV': PPV(cm),
            'NPV': NPV(cm),
            'SPC': SPC(cm),
            'SEN': SEN(cm),
            'ACC': ACC(cm),
            'cost_matrix': matrix_cost(binarized_y_true, binarized_y_pred)
        }
        test_scores.append(score)
    return pd.DataFrame(test_scores)


# if __name__ == '__main__':
#     # calc mean & std
#     df = cv_sk(indexes, clf, grid)
#     print pd.DataFrame({'mean': df.mean(), 'std': df.std()})


# from keras.models import Sequential
# from keras.layers.core import Dense, Activation

# model = Sequential()
# model.add(Dense(6, input_dim=7, init="glorot_uniform"))
# model.add(Activation("relu"))
# model.add(Dense(5, init="glorot_uniform"))
# model.add(Activation("relu"))
# model.add(Dense(4, init="glorot_uniform"))
# model.add(Activation("relu"))
# model.add(Dense(3, init="glorot_uniform"))
# model.add(Activation("softmax"))

# model.compile(loss='categorical_crossentropy', optimizer='adam')

# model.fit(X[first_nested_fold['train']],
#           pd.get_dummies(y[first_nested_fold['train']]).values,
#           batch_size=50, nb_epoch=600)
# pred = model.predict_proba(X[first_nested_fold['val']],
#                            batch_size=32)

# print 1-accuracy_score(y[first_nested_fold['val']], pred.argmax(axis=1))
