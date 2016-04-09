import numpy as np


def matrix_cost(y_true, y_pred):
    cost = np.array([[0, 2.5],
                     [5, 0]])
    result = 0
    for actual, predicted in zip(y_true, y_pred):
        result += cost[actual, predicted]
    return result
