import os
import zipfile
import numpy as np
import torch
import pandas as pd



def load_metr_la_data():

    A=np.array(pd.read_csv("",header=None).values)
    X=np.array(pd.read_csv("",header=None).values)
    X_val=np.array(pd.read_csv("",header=None).values)

    A = A.astype(np.float32)

    X = X.astype(np.float32)
    X_val = X_val.astype(np.float32)

    means = np.mean(X)
    stds = np.std(X)

    X = X - means.reshape(1, -1, 1)
    X_val = X_val - means.reshape(1, -1, 1)
    X = X / stds.reshape(1, -1, 1)
    X_val = X_val / stds.reshape(1, -1, 1)

    return A, X.transpose((1, 2, 0)), means, stds, X_val.transpose((1, 2, 0))


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """


    indices = [[int(j) for j in range(i - (num_timesteps_input*29), i+29, 29)] for i
               in range(num_timesteps_input*29, X.shape[0])]

    # Save samples
    features, target = [], []
    for i in indices:
        features.append(X[i[:-1], :, :].transpose((0, 2, 1)))
        target.append(X[i[-1:], :, :].transpose((0, 2, 1)))


    return torch.squeeze(torch.from_numpy(np.array(features)), dim=2), \
           torch.squeeze(torch.from_numpy(np.array(target)), dim=2)


def z_score(x, mean, std):
    '''
    Z-score normalization function: $z = (X - \mu) / \sigma $,
    where z is the z-score, X is the value of the element,
    $\mu$ is the population mean, and $\sigma$ is the standard deviation.
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of meadard deviation.
    :return: np.ndarray, z-score normalin.
    :param std: float, the value of stanzed array.
    '''
    return (x - mean) / std


def z_inverse(x, mean, std):
    '''
    The inverse of function z_score().
    :param x: np.ndarray, input to be recovered.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score inverse array.
    '''
    return x * std + mean


def MAPE(v, v_):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAPE averages on all elements of input.
    '''
    temp_data = np.mean(np.abs(v_ - v) / (v + 1e-5))
    return temp_data


def RMSE(v, v_):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, RMSE averages on all elements of input.
    '''
    return np.sqrt(np.mean((v_ - v) ** 2))


def MAE(v, v_):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v))

