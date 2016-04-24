import numpy as np

from settings import (
    ANN2_1_FEATURES,
    ANN2_2_FEATURES
)


def sigmoid(z):
    return 1. / (1 + np.exp(z))


def ann_2_1(df):
    X = df[ANN2_1_FEATURES].values.astype(np.float)
    a = np.matrix(X)
    weights = [
        {
            'theta': np.matrix([
                [-0.2425, 0.0782, -0.0381, 0.8974],
                [0.2636, 0.0106, 0.4893, 1.7447],
                [0.3075, 0.0001, 0.5429, 1.3764]
            ]),
            'beta': np.array([
                -1.2029, -2.2744, -1.8158
            ])
        },
        {
            'theta': np.matrix([
                [1.9384, 3.2379, 3.3631]
            ]),
            'beta': np.array([-5.4257])
        }
    ]
    for w in weights:
        a = sigmoid((a * w['theta'].T + w['beta']))

    return (np.array(a).ravel() > 0.45).astype(np.int)


def ann_2_2(df):
    X = df[ANN2_2_FEATURES].values.astype(np.float)
    a = np.matrix(X)
    weights = [
        {
            'theta': np.matrix([
                [-1.0792, 1.9383, 0.7124, -1.2664, 1.3741, 0.8298, 1.5316],
                [1.0766, 0.1376, 1.0112, -0.8320, 1.6941, 2.9541, 1.4654],
            ]),
            'beta': np.array([
                -0.5485, -1.8129
            ])
        },
        {
            'theta': np.matrix([
                [2.9753, 4.1980]
            ]),
            'beta': np.array([-3.8616])
        }
    ]
    for w in weights:
        a = sigmoid((a * w['theta'].T + w['beta']))

    return (np.array(a).ravel() > 0.60).astype(np.int)
