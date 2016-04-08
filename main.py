from collections import OrderedDict

import yaml
import pandas as pd
import numpy as np

from nested_kfold import nested_kfold
from ladder.run import train_own_dataset
from fuel.datasets import IndexableDataset

LADDERS_CONFIG = 'ladder_models.yaml'
DATASET = 'data/dataset.csv'
TARGET_NAME = 'MalignancyCharacter'


with open(LADDERS_CONFIG, 'r') as fp:
    configs = yaml.load(fp)


df = pd.read_csv(DATASET)
y = df[TARGET_NAME].values.astype(np.int)
indexes = nested_kfold(y, method='stratified')
first_fold = indexes[0]
first_nested_fold = first_fold['nested_indexes'][0]


class OvaDataset(IndexableDataset):
    def __init__(self, X, y, **kwargs):
        indexables = OrderedDict(
            [('features', X),
             ('targets', y)]
        )
        super(OvaDataset, self).__init__(indexables, **kwargs)


for name, config in configs.iteritems():
    X = df[config.pop('x_features')].values.astype(np.float)
    train_own_dataset(
        config,
        dataset={
            'ovadataset': OvaDataset(X, y),
            'train_indexes': first_nested_fold['train'],
            'val_indexes': first_nested_fold['val'],
        }
    )
