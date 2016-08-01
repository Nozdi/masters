#!/usr/bin/env python
import argparse

import pandas as pd
import numpy as np

import theano.tensor as T
from keras.models import model_from_yaml

RESULTS_DIRECTORY = './results/dl'

X_FEATURES = ['Color', 'Ca125', 'AgeAfterMenopause']


def weighted_cost(y_true, y_pred):
    benign_mask = T.eq(y_true[:, 0], 1).nonzero()[0]
    malignant_mask = T.neq(y_true[:, 0], 1).nonzero()[0]
    return T.concatenate(
        [T.nnet.categorical_crossentropy(
            y_pred[benign_mask], y_true[benign_mask]),
         2 * T.nnet.categorical_crossentropy(
            y_pred[malignant_mask], y_true[malignant_mask])]
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script which predicts malignancy using given features')
    parser.add_argument(
        'Color', type=int,
        help='Color - IOTA Amount of blood flow 1 / 2 / 3 / 4')
    parser.add_argument(
        'Ca125', type=float,
        help='Ca125 - The blood serum marker 0 - 1500')
    parser.add_argument(
        'AgeAfterMenopause', type=int,
        help='AgeAfterMenopause - how many years after menopause'
             ' (0 if menopause didn\'t occurred)'
    )

    args = parser.parse_args()

    with open('{}/model_config.yaml'.format(RESULTS_DIRECTORY)) as fp:
        model = model_from_yaml(fp.read())

    model.load_weights('{}/model_weights.h5'.format(RESULTS_DIRECTORY))

    model.compile(loss=weighted_cost, optimizer='adam')

    proba_prediction = model.predict(
        np.array([[getattr(args, feature) for feature in X_FEATURES]])
    )
    classes = ['Benign', 'Malignant']
    result = pd.DataFrame(
        proba_prediction,
        columns=map(lambda c: '{} Probability'.format(c), classes)
    )
    result['Prediction'] = classes[proba_prediction.argmax()]
    print result.to_string(index=False)
