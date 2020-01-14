import numbers
import numpy as np

# numpy-ml/numpy_ml/utils/testing.py/

#######################################################################
#                           Data Generators                           #
#######################################################################


def random_one_hot_matrix(n_examples, n_classes):
    """Create a random one-hot matrix of shape (`n_examples`, `n_classes`)"""
    X = np.eye(n_classes)
    X = X[np.random.choice(n_classes, n_examples)]
    return X


def random_stochastic_matrix(n_examples, n_classes):
    """Create a random stochastic matrix of shape (`n_examples`, `n_classes`)"""
    X = np.random.rand(n_examples, n_classes)
    X /= X.sum(axis=1, keepdims=True)
    return X


def random_tensor(shape, standardize=False):
    """
    Create a random real-valued tensor of shape `shape`. If `standardize` is
    True, ensure each column has mean 0 and std 1.
    """
    offset = np.random.randint(-300, 300, shape)
    X = np.random.rand(*shape) + offset

    if standardize:
        eps = np.finfo(float).eps
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + eps)
    return X



def gaussian_cdf(x, mean, var):
    """
    Compute the probability that a random draw from a 1D Gaussian with mean
    `mean` and variance `var` is less than or equal to `x`.
    """
    eps = np.finfo(float).eps
    x_scaled = (x - mean) / np.sqrt(var + eps)
    return (1 + erf(x_scaled / np.sqrt(2))) / 2
    
#######################################################################
#                          functions Metal                            #
#######################################################################


#######################################################################
#                           ML-scratch                                #
#######################################################################
