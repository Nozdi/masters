# coding: utf-8
import numpy as np
from sklearn.metrics import confusion_matrix as calc_cm


def binarize_y(y):
    return ((y == 1) | (y == 2)).astype(np.int)


def matrix_cost(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    cost = np.array([[0, 2.5],
                     [5, 0]])
    return cost[y_true, y_pred].sum()


def PPV(confusion_matrix):
    # precision
    TP = confusion_matrix[1, 1]
    FP = confusion_matrix[0, 1]
    return TP / (TP + FP)


def NPV(confusion_matrix):
    TN = confusion_matrix[0, 0]
    FN = confusion_matrix[1, 0]
    return TN / (TN + FN)


def SPC(confusion_matrix):
    FP = confusion_matrix[0, 1]
    TN = confusion_matrix[0, 0]
    return TN / (TN + FP)


def SEN(confusion_matrix):
    # recall
    TP = confusion_matrix[1, 1]
    FN = confusion_matrix[1, 0]
    return TP / (TP + FN)


def ACC(confusion_matrix):
    TP = confusion_matrix[1, 1]
    FP = confusion_matrix[0, 1]
    TN = confusion_matrix[0, 0]
    FN = confusion_matrix[1, 0]
    return (TP + TN) / (TP + FP + TN + FN)


def create_score_dict(y_true, y_pred):
    cm = calc_cm(y_true, y_pred).astype(np.float)
    return {
        'PPV': PPV(cm),
        'NPV': NPV(cm),
        'SPC': SPC(cm),
        'SEN': SEN(cm),
        'ACC': ACC(cm),
        'cost_matrix': matrix_cost(y_true, y_pred)
    }
