from sklearn.cross_validation import KFold


SEED = 1
N = 200


def nested_kfold(n=N, n_folds=10, shuffle=True, random_state=SEED):
    k_fold = KFold(n, n_folds=n_folds, shuffle=shuffle, random_state=random_state)
    all_indexes = []
    for train_indices, test_indices in k_fold:
        result = {}
        nested_fold = KFold(len(train_indices), n_folds=n_folds,
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
