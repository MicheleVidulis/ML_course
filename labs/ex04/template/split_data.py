# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    # Shuffle and sample
    N = len(y)
    shuffle_indices = np.random.permutation(np.arange(N))
    shuffled_x = x[shuffle_indices]
    shuffled_y = y[shuffle_indices]
    training_instances_number = int(np.floor(N*ratio))
    tx_train = shuffled_x[0:training_instances_number]
    y_train = shuffled_y[0:training_instances_number]
    tx_test = shuffled_x[training_instances_number+1:-1]
    y_test = shuffled_y[training_instances_number+1:-1]
    return tx_train, y_train, tx_test, y_test
