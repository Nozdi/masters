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
    # train_own_dataset(
    #     config,
    #     dataset={
    #         'ovadataset': OvaDataset(X, y),
    #         'train_indexes': first_nested_fold['train'],
    #         'val_indexes': first_nested_fold['val'],
    #     }
    # )


# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score


# clf = make_pipeline(
#     StandardScaler(),
#     LogisticRegression(multi_class='multinomial', solver='lbfgs')
# )

# clf.fit(X[first_nested_fold['train']], y[first_nested_fold['train']])
# print 1 - accuracy_score(
#     y[first_nested_fold['val']],
#     clf.predict(X[first_nested_fold['val']])
# )

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
