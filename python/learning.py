#!/usr/bin/env python

import itertools
import numpy as np
import kmc2  # state-of-the-art kmeans initialization (as of NIPS 2016)
from sklearn import cluster
from .utils import sliding_window as window


# ================================================================ kmeans

def kmeans(X, k, max_iter=16, init='kmc2'):
    X = X.astype(np.float32)
    np.random.seed(123)

    # if k is huge, initialize centers with cartesian product of centroids
    # in two subspaces
    if init == 'subspaces':
        sqrt_k = int(np.sqrt(k) + .5)
        if sqrt_k ** 2 != k:
            raise ValueError("K must be a square number if init='subspaces'")

        _, D = X.shape
        centroids0, _ = kmeans(X[:, :D/2], sqrt_k, max_iter=2)
        centroids1, _ = kmeans(X[:, D/2:], sqrt_k, max_iter=2)
        seeds = np.empty((k, D), dtype=np.float32)
        for i in range(sqrt_k):
            for j in range(sqrt_k):
                row = i * sqrt_k + j
                seeds[row, :D/2] = centroids0[i]
                seeds[row, D/2:] = centroids1[j]

    elif init == 'kmc2':
        seeds = kmc2.kmc2(X, k).astype(np.float32)
    else:
        raise ValueError("init parameter must be one of {'kmc2', 'subspaces'}")

    estimator = cluster.MiniBatchKMeans(k, init=seeds, max_iter=max_iter).fit(X)
    return estimator.cluster_centers_, estimator.labels_


# ================================================================ misc

def least_squares(A, y):
    """Returns x, err such that err = ||Ax - y||^2 is minimized"""
    assert A.shape[0] == len(y)
    x_ideal, residual, _, _ = np.linalg.lstsq(A, y.ravel())
    return x_ideal


# ================================================================

def learn_filters(x, ntaps=4, nfilters=16, niters=8, quantize=False,
                  init='brute', verbose=1):
    x = np.asarray(x, dtype=np.float32).ravel()
    N = len(x) - ntaps

    X = window.sliding_window_1D(x[:-1], ntaps).astype(np.float32)
    y = x[ntaps:].astype(np.float32).reshape((-1, 1))  # col vect
    # assert len(X) == len(y)

    filters = np.random.randn(nfilters, ntaps).astype(np.float32)
    # first filter just predicts previous value (delta coding)
    filters[0, :] = 0
    filters[0, -1] = 1
    # second filter does double delta encoding
    filters[1, :] = 0
    filters[1, -1] = 2
    filters[1, -2] = -1
    # third filter low-pass filter as initialization ?
    filters[2, :] = 1. / ntaps

    initial_filters = np.copy(filters)
    y_var = np.var(y)

    if init == 'brute':
        pass # TODO

    for it in range(niters):
        predictions = np.dot(X, filters.T)
        errs = np.abs(predictions - y)
        assigs = np.argmin(errs, axis=1)

        if verbose > 1:
            best_errs = np.min(errs, axis=1)
            # print "{}: MSE / var(y) = {:.04f}; best err^2 / var(y) = {:.04f}".format(
            #     it, np.mean(errs*errs) / y_var,
            #     np.mean(best_errs*best_errs) / y_var)
            print "{}: MSE / var(y) = {:.04f}".format(
                it, np.mean(best_errs*best_errs) / y_var)
            print "    bincounts: ", np.bincount(assigs) / float(N)

        # for i in range(1, nfilters):  # skip first filter, which is fixed
        min_occurrences = int((N / float(nfilters)) / 4)
        for i in range(2, nfilters):  # skip first 2 filters, which are fixed
            X_filt = X[assigs == i]
            y_filt = y[assigs == i]
            if len(y_filt) < min_occurrences:
                filters[i] = np.random.randn(ntaps)  # random restart
            else:
                filters[i] = least_squares(X_filt, y_filt)

    if verbose > 1:
        print "final filters:\n", filters

    if verbose > 0:
        initial_predictions = np.dot(X, initial_filters.T)
        initial_errs = initial_predictions - y
        initial_best_errs = np.min(initial_errs * initial_errs, axis=1)
        predictions = np.dot(X, filters.T)
        errs = predictions - y
        best_errs = np.min(errs * errs, axis=1)
        print "initial, final MSE / var(y) = {:.04f}, {:.04f}".format(
            np.mean(initial_best_errs) / y_var, np.mean(best_errs) / y_var)

        print "    bincounts: ", np.bincount(assigs) / float(N)

    return filters


def all_possible_filters(ntaps, nbits=4, step_sz=.25):
    assert (1 << nbits) ** ntaps < 100*1000  # ensure tractable computation

    nvals = 1 << nbits
    possible_vals = np.arange(nvals, dtype=np.float32)
    possible_vals += int(1. / step_sz) - (nvals >> 1)  # center at +1
    possible_vals *= step_sz

    vals_list = [possible_vals for i in range(ntaps)]  # ntaps copies of vals

    all_vals = itertools.product(*vals_list)
    return np.array(list(all_vals))  # intermediate list so array ctor works


def prediction_errors(X, y, filters):
    return y - np.dot(X, filters.T)


# def sq_prediction_errors(X, y, filters):
#     predictions = np.dot(X, filters.T)
#     errs = y - predictions
#     # print "errs.shape", errs.shape
#     # print "sq errs.shape", (errs * errs).shape
#     return errs * errs
#     # if loss == 'l2':
#     #     return errs * errs
#     # elif loss == 'l1':
#     #     return np.abs(errs)
#     # elif loss == 'linf':
#     #     return np.max(errs, axis=-1)


# def min_sq_errs(X, y, filters, return_which_filter=False):
#     # predictions = np.dot(X, filters.T)
#     # errs = y - predictions
#     # errs = errs * errs
#     errs = sq_prediction_errors(X, y, filters)

#     min_errs = np.min(errs, axis=1)
#     if return_which_filter:
#         assigs = np.argmin(errs, axis=1)
#         return min_errs, assigs

#     return min_errs


def trim_to_multiple_of(x, divisor):
    tail = len(x) % divisor
    return x[:-tail] if tail > 0 else x


# def group_into_blocks(X, block_sz):
#     """Reshapes `X` such that its first dimension is divided by `block_sz`,
#     and it gets a third dimension of size `block_sz`.

#     Note that this method will discard rows of `X` until
#     `X.shape[0] % block_sz == 0`.

#     shape [a, b] -> [a / block_sz, b, block_sz]  # 2D X
#     shape [a] -> [a / block_sz, 1, block_sz]     # 1D X
#     """
#     # if block_sz < 2:
#     #     return X.reshape((len(X), -1, 1))
#     block_sz = max(block_sz, 1)
#     if len(X.shape) == 1:
#         X = X.reshape(-1, 1)
#     X = trim_to_multiple_of(X, block_sz)
#     assert len(X) % block_sz == 0
#     return X.reshape((X.shape[0] / block_sz, X.shape[1], block_sz))

def windows_as_dim3(X, windowlen, stride=1):
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    assert len(X.shape) == 2
    winshape = (windowlen, X.shape[1])
    stride = (stride, 1)  # 2nd value irrelevant
    return window.sliding_window(X, winshape, stride)


def compute_loss(errs, loss='l2', axis=-1):
    """sums along last axis; errs should be y - y_hat"""
    if loss == 'l2':
        return (errs * errs).sum(axis=axis)
    elif loss == 'l1':
        return np.abs(errs).sum(axis=axis)
    elif loss == 'linf':
        return np.max(np.abs(errs), axis=axis)
    else:
        raise ValueError("Unrecognized loss function '{}'".format(loss))




def learn_kmeans(x, k=16, ntaps=4, loss='l2', verbose=1):
    pass


    # SELF: pick up here





def greedy_brute_filters(x, nfilters=16, ntaps=4, nbits=4, step_sz=.5,
                         block_sz=-1, loss='l2', verbose=1):
    """
    Assess every possible filter and greedily take the best

    Args:
        nfilters (int): number of filters to learn
        ntaps (int, optional): length of each filter
        nbits (int, optional): bit depth of the filters
        step_sz (float, optional): difference between two values for a given
            tap (ie, how the integer bit representation translates to floats)
        block_sz (int, optional): number of successive values to train
            each filter to predict. Block size of 1 (the default) corresponds
            to learning filters that predict the next value
        loss (string): see `compute_loss()`
        verbose (int, optional): 1 to log final MSE improvement; 2 to log
            everything

    Returns:
        filters (nfilters x ntaps array of float32): the learned filters. Note
        that the first filter always implements delta coding and the second
        filter always implements double delta coding.
    """
    block_sz = max(1, block_sz)

    x = np.asarray(x, dtype=np.float32).ravel()
    # N = len(x) - ntaps
    X = window.sliding_window_1D(x[:-1], ntaps).astype(np.float32)
    y = x[ntaps:].astype(np.float32).reshape((-1, 1))  # col vect

    # filters = np.zeros((2, ntaps))
    filters = np.zeros((2, ntaps))
    # filters = np.zeros((3, ntaps))
    # filters[0, -1] = 1  # delta encoding
    # filters[1, -1], filters[1, -2] = 2, -1  # delta-delta encoding
    # filters[2, -1], filters[2, -2], filters[2, -3] = 2, -3, 1  # 3delta
    filters[1, -1] = 1  # delta encoding as second thing (first thing all 0s)

    candidates = all_possible_filters(ntaps, nbits, step_sz)

    # print "candidates.shape: ", candidates.shape
    # print "supposed ncandidates: ", int((1 << nbits) ** ntaps)
    assert candidates.shape[0] == int((1 << nbits) ** ntaps)
    assert candidates.shape[1] == ntaps

    # errors using initial filters (delta and double delta)
    errs = prediction_errors(X, y, filters)
    # errors from all candidate filters
    # all_errs = min_sq_errs(X, y, candidates)  # N x len(candidates)
    all_errs = prediction_errors(X, y, candidates)  # N x len(candidates)

    # print "initial errs shape: ", errs.shape
    # print "initial all_errs shape: ", all_errs.shape

    # if using blocks of elements for the errors, we pick filters that
    # reduce the error for entire blocks, not individual samples; this
    # is important because in the real algorithm, a filter is selected
    # to predict the next 8 points, not just 1
    if block_sz > 1:
        # y = windows_as_dim3(y, block_sz)
        errs = windows_as_dim3(errs, block_sz)
        all_errs = windows_as_dim3(all_errs, block_sz)
        y = y[:len(errs)]  # we don't use y later, but this is the right length
    else:
        errs = errs[..., np.newaxis]
        all_errs = all_errs[..., np.newaxis]

    # print "raw errs shape: ", errs.shape
    # print "raw all_errs shape: ", all_errs.shape

    # compute loss, reduced across block dimension (which is dim 1, not 2)
    axis = 1 if block_sz > 1 else -1
    errs = compute_loss(errs, loss, axis=axis)
    all_errs = compute_loss(all_errs, loss, axis=axis)

    # take the best error attainable using the filters we have so far
    errs = np.min(errs, axis=1).reshape(-1, 1)  # N x 1

    # print "y shape: ", y.shape
    # print "errs shape: ", errs.shape
    # print "all_errs shape: ", all_errs.shape

    assert all_errs.shape == (errs.shape[0], len(candidates))

    if verbose > 0:
        if loss == 'l2':
            div_by = np.var(y) * block_sz
        elif loss == 'l1':
            div_by = np.mean(np.abs(y)) * block_sz
        elif loss == 'linf':
            div_by = np.mean(np.abs(y))
        initial_mean_err = np.mean(errs)

    for i in range(filters.shape[0], nfilters):
        min_errs = np.minimum(all_errs, errs)  # N x len(candidates)
        losses = np.mean(min_errs, axis=0)  # total err from using each filt
        best_filt_idx = np.argmin(losses)
        # add best filter
        filters = np.vstack((filters, candidates[best_filt_idx]))
        errs = min_errs[:, best_filt_idx].reshape(y.shape)

        if verbose > 1:
            print "{}: mean err / var: {}".format(
                i, losses[best_filt_idx] / div_by)

    if verbose > 0:
        final_mean_err = np.mean(errs)
        print "-> initial err, final err: {:.5f}, {:.5f}".format(
                initial_mean_err / div_by, final_mean_err / div_by)

        y = x[ntaps:].astype(np.float32).reshape((-1, 1))  # col vect
        raw_errs = y - np.dot(X, filters.T)
        losses = raw_errs * raw_errs
        assigs = np.argmin(losses, axis=1)
        print "    bincounts (ignoring blocks): {}".format(
            np.bincount(assigs) / float(len(X)))

        if block_sz > 1:
            block_losses = windows_as_dim3(losses, block_sz).sum(axis=1)
            assigs = np.argmin(block_losses, axis=1)
            print "    bincounts (using blocks): {}".format(
                np.bincount(assigs) / float(len(X)))

        if verbose > 1:
            print "final filters:\n", filters

    assert filters.shape == (nfilters, ntaps)
    return filters


def main():
    np.set_printoptions(formatter={'float': lambda x: '{:.3f}'.format(x)})

    # A = np.arange(15).reshape(5, 3)
    # winshape = (2, A.shape[1])
    # stride = (1, 1)
    # ret = window.sliding_window(A, winshape, stride)
    # print "window shape: ", ret.shape
    # print "windows:\n", ret
    # print "windows max:\n", ret.max(axis=-1)
    # return

    x = np.random.randn(100)
    # learn_filters(x, nfilters=2)  # just delta and double delta
    # learn_filters(x, nfilters=8, niters=10)  # just delta and double delta
    # learn_filters(x)

    nbits = 2
    nbits = 3
    # step_sz = .25
    step_sz = .5
    # step_sz = 1
    # block_sz = 1
    block_sz = 4

    greedy_brute_filters(x, nfilters=8, ntaps=4, nbits=nbits,
                     block_sz=block_sz, step_sz=step_sz, verbose=2)

    # print "done"


if __name__ == '__main__':
    main()
