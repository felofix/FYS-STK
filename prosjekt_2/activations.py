import numpy as np
from autograd import elementwise_grad

def identity(X):
    return X

def sigmoid(X):
    return 1/(1 + np.exp(-X))

def RELU(X):
    return np.where(X > np.zeros(X.shape), X, np.zeros(X.shape))

def LRELU(X):
    delta = 10e-4
    return np.where(X > np.zeros(X.shape), X, delta * X)
