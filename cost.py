import numpy as np


def matrix_cost(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    cost = np.array([[0, 2.5, 2.5],
                     [5, 0, 0],
                     [5, 0, 0]])
    result = 0
    for actual, predicted in zip(y_true, y_pred):
        result += cost[actual, predicted]
    return result
