import json
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


def weighted_cost(y_true, y_pred):
    multiplier = y_true[:, 0] + y_true[:, 1] * 2
    return T.mean(
        multiplier * T.nnet.binary_crossentropy(y_pred[:, 1], y_true[:, 1])
    )


ann_grid = {
    'batch_size': [25, 50],
    'nb_epoch': range(200, 501, 20),
    'optimizer': ['sgd', 'adam'],
    'output_dim': [2]
}

x_features = ['Color', 'Ca125', 'APapDimension',
              'AgeAfterMenopause', 'PapBloodFlow']
target = 'MalignancyCharacter'

optimizers = {
    'sgd': 'sgd',
    'sgd_paramterized': SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True),
    'adam': 'adam',
    'rmsprop': 'rmsprop',
}
n_cores = 20
dataset_filepath = 'data/dataset.csv'
df = pd.read_csv(dataset_filepath)
y = df[target].values.astype(np.int)
y_bin = binarize_y(y)
indexes = nested_kfold(y, method='stratified')


def prepare_ann(optimizer, input_dim, output_dim=3):
    np.random.seed(seed)  # for reproducibility
    optimizer = optimizers.get(optimizer)
    model = Sequential()
    model.add(BatchNormalization(input_shape=(input_dim, )))
    model.add(Dropout(0.2))
    model.add(Dense(5, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss=weighted_cost, optimizer=optimizer)
    return model


def pred_ann(
    df, train_idx, val_idx, config,
    feature_list=x_features, target=target
):
    output_dim = config['output_dim']
    model = prepare_ann(
        config['optimizer'],
        input_dim=len(feature_list), output_dim=output_dim)

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


def cv_nn(indexes, grid, pred_function=pred_ann, df=df, y=y, name='simple_dl_10'):
    test_scores = []
    for idx, fold in enumerate(indexes):
        configs = list(ParameterGrid(grid))
        nested_cv_results = Parallel(n_jobs=n_cores)(
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
            "./results/fold_{}_{}_scores.csv".format(name, idx), index=False
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
            create_score_dict(y_bin[fold['test']], pred)
        )
        print test_scores[-1]
    results = pd.DataFrame(test_scores)
    results.to_csv("./results/{}_all.csv".format(name), index=False)
    return results


if __name__ == '__main__':
    df = cv_nn(indexes, ann_grid)
    print df
    print df.mean()
    print df.std()
