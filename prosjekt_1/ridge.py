# Importing packages.
import numpy as np

def mean_scale(Xs):
    # Scaling function.
    n = X.shape[1]
    
    for i in range(n):
        avg = np.mean(Xs[:, i])
        Xs[i, :] -= avg

    return Xs

def calculate_beta_ridge(X_train, y_train, lamb):
    # Calculating beta values with ridge regression. 
    beta = np.linalg.inv(X_train.T @ X_train + np.eye(X_train.shape[1])*lamb) @ X_train.T @ y_train
    return beta

