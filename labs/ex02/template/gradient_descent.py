# -*- coding: utf-8 -*-
"""Gradient Descent"""

from costs import compute_loss
import numpy as np

def compute_gradient(y, tx, w, loss_type):
    """Compute the gradient."""
    N = len(y)
    e = y - tx.dot(w)
    
    if loss_type == "MSE":
        gradient = -1/N * tx.transpose().dot(e)
        
    elif loss_type == "MAE":
        gradient = np.zeros(len(w))
        subgrad = np.zeros(N)
        tol = np.max(e)/10
        for n in range(N):
            if e[n] < -tol:
                subgrad[n] = -1
            elif e[n] > tol:
                subgrad[n] = 1
            else:
                subgrad[n] = 0.9
        gradient = -1/N * tx.T.dot(subgrad)
    
    return gradient


def gradient_descent(y, tx, initial_w, max_iters, gamma, loss_type):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        # Compute gradient and loss
        grad = compute_gradient(y, tx, w, loss_type)
        loss = compute_loss(y, tx, w, loss_type)
        # Update w by gradient
        w = w - gamma*grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        
    return losses, ws