# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

from helpers import batch_iter
from costs import compute_loss

def compute_stoch_gradient(y, tx, w, loss_type):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    batch_size = len(y)
    e = y - tx.dot(w)
    
    if loss_type == "MSE":
        gradient = -1/batch_size * tx.transpose().dot(e)
        
    elif loss_type == "MAE":
        gradient = np.zeros(len(w))
        subgrad = np.zeros(batch_size)
        tol = np.max(e)/10
        for n in range(batch_size):
            if e[n] < -tol:
                subgrad[n] = -1
            elif e[n] > tol:
                subgrad[n] = 1
            else:
                subgrad[n] = 0.9
        gradient = -1/N * tx.T.dot(subgrad)
    
    return gradient


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma, loss_type):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    batch_size = 5
    
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            grad = compute_stoch_gradient(minibatch_y, minibatch_tx, w, loss_type)
            loss = compute_loss(y, tx, w, loss_type)
            # Update
            w = w - gamma*grad
            # Store
            ws.append(w)
            losses.append(loss)
            
        print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
        
    return losses, ws