import numpy as np
import pandas as pd

### Splitters
def split_failure_data(failure_data, train_cell_ids):
    """
    Helper function to split failure data into train/test sets for each independent set. Matches cell_ids to the
    corresponding rows in xy_pairs and splits into train/test based on whether cell_id is in cell_ids_train.

    Parameters:
        train_cell_ids (list): List of cell_ids to use for training. Should be a subset of the unique cell_ids across all sets.
        failure_data (tuple): Tuple of (cell_ids, xy_pairs) where cell_ids is a list of lists of cell_ids for each set,
                                and xy_pairs is a list of tuples (X, Y) for each set.

    Returns:
        (training failure data, test failure data): Each is a tuple of (cell_ids, xy_pairs) similar to the input
    """
    train_xy_pairs = []
    train_cell_ids_out = []
    test_xy_pairs = []
    test_cell_ids = []

    cell_ids, xy_pairs = failure_data
    for cell_id_set, (X, Y) in zip(cell_ids, xy_pairs):
        cell_id_set = pd.Series(cell_id_set)
        mask_train = cell_id_set.isin(train_cell_ids)
        mask_test = ~mask_train

        if mask_train.any():
            X_train = X[mask_train]
            Y_train = Y[mask_train]
            train_xy_pairs.append((X_train, Y_train))
            train_cell_ids_out.append(cell_id_set[mask_train].to_list())
        if mask_test.any():
            X_test = X[mask_test]
            Y_test = Y[mask_test]
            test_xy_pairs.append((X_test, Y_test))
            test_cell_ids.append(cell_id_set[mask_test].to_list())

    return (train_cell_ids_out, train_xy_pairs), (test_cell_ids, test_xy_pairs)

def split_rul_data(rul_data, train_cell_ids):
    """
    Helper function to split RUL data into train/test sets for each independent set. Matches cell_ids to the
    corresponding rows in xy_pairs and splits into train/test based on whether cell_id is in train_cell_ids.

    Parameters:
        rul_data (tuple): Tuple of (cell_ids, xy_pairs) where cell_ids is a list of cell_ids,
                          and xy_pairs is a tuple of X and Y matrices.
        train_cell_ids (list): List of cell_ids to use for training. Should be a subset of the unique cell_ids across all sets.

    Returns:
        (training failure data, test failure data): Each is a tuple of (cell_ids, xy_pairs) similar to the input
    """
    cell_ids, (x, y) = rul_data
    cell_ids = pd.Series(cell_ids)

    cell_ids_train = cell_ids[cell_ids.isin(train_cell_ids)].to_list()
    x_train = x[cell_ids.isin(train_cell_ids)].reset_index(drop=True)
    y_train = y[cell_ids.isin(train_cell_ids)].reset_index(drop=True)

    cell_ids_test = cell_ids[~cell_ids.isin(train_cell_ids)].to_list()
    x_test = x[~cell_ids.isin(train_cell_ids)].reset_index(drop=True)
    y_test = y[~cell_ids.isin(train_cell_ids)].reset_index(drop=True)

    return (cell_ids_train, (x_train, y_train)), (cell_ids_test, (x_test, y_test))