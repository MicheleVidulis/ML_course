# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    N = len(y)
    lambda_prime = lambda_*(2*N)
    weights = np.linalg.solve(tx.T @ tx + lambda_prime*np.eye(tx.shape[1]), tx.T @ y)
    e = y - tx@weights
    mse = 1/(2*N) * (e.T @ e)
    return mse, weights
