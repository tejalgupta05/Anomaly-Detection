from numpy.linalg import det, pinv
from math import sqrt, pi
import numpy as np


def estimateGaussian(X):
    mean = np.mean(X, axis=0)
    variance = np.var(X, axis=0)

    return mean, variance


def multivariateGaussian(X, mu, sigma):
    k = len(mu)

    sigma = np.diag(sigma)

    """X = X - mu.reshape((1, X.shape[1]))
    temp = X.dot(pinv(sigma))
    temp = temp.dot(X.T)
    #print((2*np.pi)**(-k/2) * np.exp([-0.5 * -9.83391146e-12]))
    #print(np.exp(-0.5 * np.sum(temp, axis=1)))

    p = (2*np.pi)**(-k/2) * (det(sigma)**(-0.5)) * np.exp(-0.5 * np.sum(temp, axis=1))
    """
    p = (2*np.pi)**(-k/2) * (det(sigma)**(-0.5)) * np.exp(-0.5 * np.sum(X @ pinv(sigma) * X, axis=1))

    return p


def selectThreshold(p, y):
    step = (np.max(p) - np.min(p)) / 1000

    bestEpsilon = 0
    bestF1 = 0
    F1 = 0

    epsilon = np.min(p)
    while epsilon < np.max(p):
        predictions = np.where(p < epsilon, 1, 0)

        tp = 0
        fp = 0
        fn = 0

        for i in range(len(y)):
            if y[i] == 1 and predictions[i] == 1:
                tp = tp + 1

            elif y[i] == 0 and predictions[i] == 1:
                fp = fp + 1

            elif y[i] == 1 and predictions[i] == 0:
                fn = fn + 1

        if tp+fp != 0 and tp+fn != 0:
            precision  = tp / (tp + fp)
            recall = tp / (tp + fn)

            if precision != 0 and recall != 0:
                F1 = 2 * precision * recall / (precision + recall)

            else:
                epsilon += step
                continue

        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon

        epsilon += step

    return bestEpsilon, bestF1
