# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

import numpy as np

def compute_loss(y, tx, w, loss_type):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE / MAE
    # ***************************************************
    
#     MSE = True        # MSE --> True, MAE --> False
    N = len(y)

    if loss_type == "MSE":
        L = 1/(2*N) * np.sum((y - np.dot(tx,w))**2)
    elif loss_type == "MAE":
        L = 1/(N) * np.sum(np.abs(y - np.dot(tx,w)))
        
    return L

    