# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares solution."""
    N = len(y)
    weights = np.linalg.solve(tx.T @ tx, tx.T @ y)
    e = y - tx@weights
    mse = 1/(2*N) * (e.T @ e)
    return mse, weights