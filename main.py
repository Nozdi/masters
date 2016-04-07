import yaml
import pandas as pd
import numpy as np

from nested_kfold import nested_kfold
from ladder import train_own_dataset


LADDERS_CONFIG = 'ladder_models.yaml'
DATASET = 'data/dataset.csv'
TARGET_NAME = 'MalignancyCharacter'


with open(LADDERS_CONFIG, 'r') as fp:
    configs = yaml.load(fp)


df = pd.read_csv(DATASET)
y = df[TARGET_NAME].values.astype(np.float)
indexes = nested_kfold(y, method='stratified')
first_fold = indexes[0]
first_nested_fold = first_fold['nested_indexes'][0]


for name, config in configs.iteritems():
    X = df[config.pop('x_features')].values.astype(np.float)

    train_X = X[first_nested_fold['train']]
    train_Y = y[first_nested_fold['train']]

    val_X = X[first_nested_fold['val']]
    val_Y = y[first_nested_fold['val']]

    train_own_dataset(
        config,
        dataset={
            'train_X': train_X,
            'train_Y': train_Y,
            'val_X': val_X,
            'val_Y': val_Y,
        }
    )
