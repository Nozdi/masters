import numpy as np

from sklearn.cross_validation import (
    KFold,
    StratifiedKFold,
)

SEED = 1
METHODS = {
    'standard': KFold,
    'stratified': StratifiedKFold
}


def nested_kfold(y, n_folds=10, shuffle=True,
                 random_state=SEED, method='standard'):
    """
    y - array of classes
    """
    folding_class = METHODS.get(method)
    n = y
    if method == 'standard':
        n = len(y)
    k_fold = folding_class(n, n_folds=n_folds, shuffle=shuffle,
                           random_state=random_state)
    all_indexes = []
    for train_indices, test_indices in k_fold:
        result = {}
        nested_n = nested_y = y[train_indices]
        if method == 'standard':
            nested_n = len(nested_y)

        nested_fold = folding_class(nested_n, n_folds=n_folds,
                                    shuffle=shuffle, random_state=random_state)

        nested_indexes = []
        for nested_train, nested_val in nested_fold:
            current = {}
            current['train'] = train_indices[nested_train]
            current['val'] = train_indices[nested_val]
            nested_indexes.append(current)
        result['train'] = train_indices
        result['test'] = test_indices
        result['nested_indexes'] = nested_indexes
        all_indexes.append(result)
    return all_indexes


if __name__ == '__main__':
    y = np.array([np.random.randint(2) for _ in range(100)])
    print nested_kfold(y)
    print nested_kfold(y, method='stratified')
