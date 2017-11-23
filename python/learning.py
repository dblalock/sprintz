#!/usr/bin/env python

from __future__ import division

import itertools
import numpy as np
from sklearn import linear_model as linear  # for VAR

from .utils import sliding_window as window
# from .utils.distance import kmeans, dists_sq
from .utils import distance as dist
from . import compress


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
    if len(filters):
        return y - np.dot(X, filters.T)
    # return y - np.dot(X, np.zeros((X.shape[1], 1)))
    return y.reshape(len(y), 1)

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

    filters = []
    # filters = np.zeros((1, ntaps))
    # filters = np.zeros((2, ntaps))
    # filters = np.zeros((2, ntaps))
    # filters = np.zeros((3, ntaps))
    # filters = np.zeros((4, ntaps))
    # filters[0, -1] = 1  # delta encoding
    # filters[1, -1], filters[1, -2] = 2, -1  # delta-delta encoding
    # filters[2, -1], filters[2, -2], filters[2, -3] = 2, -3, 1  # 3delta
    # filters[2, -1], filters[2, -2], filters[2, -3] = .5, .5, 0  # lpf
    # filters[3, -1], filters[3, -2], filters[2, -3] = 1, -1, 0  # alternating delta
    # filters[1, -1] = 1  # delta encoding as second thing (first thing all 0s)

    if len(filters) == nfilters:
        return filters

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
    losses = compute_loss(errs, loss, axis=axis)
    all_losses = compute_loss(all_errs, loss, axis=axis)

    # print "errs.shape: ", errs.shape
    # return

    # print "y shape: ", y.shape
    # print "losses shape: ", errs.shape
    # print "all_losses shape: ", all_errs.shape
    # return

    # take the best loss attainable using the filters we have so far
    if len(losses.shape) < 2:  # true iff we start with < 2 filters
        assert len(filters) in (0, 1)
        losses = losses.reshape(-1, 1)
    else:
        losses = np.min(losses, axis=1).reshape(-1, 1)  # N x 1

    assert all_losses.shape == (errs.shape[0], len(candidates))

    if verbose > 0:
        if loss == 'l2':
            div_by = np.var(y) * block_sz
        elif loss == 'l1':
            div_by = np.mean(np.abs(y)) * block_sz
        elif loss == 'linf':
            div_by = np.mean(np.abs(y))
        initial_mean_err = np.mean(losses)

    for i in range(len(filters), nfilters):
        min_losses = np.minimum(all_losses, losses)  # N x len(candidates)
        mean_losses = np.mean(min_losses, axis=0)  # total err from using each filt
        best_filt_idx = np.argmin(mean_losses)
        # add best filter
        if len(filters):
            filters = np.vstack((filters, candidates[best_filt_idx]))
        else:
            filters = candidates[best_filt_idx]
        losses = min_losses[:, best_filt_idx].reshape(y.shape)  # make col vect

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

    if len(filters.shape) == 1:
        filters = filters.reshape(1, len(filters))
    assert filters.shape == (nfilters, ntaps)
    return filters


# ================================================================ kmeans

# ------------------------------------------------ offline kmeans

# ideal case; offline kmeans on the data we're compressing
def sub_kmeans(blocks, k=16, verbose=1):
    centroids, assigs = dist.kmeans(blocks, k=k)
    centroids = centroids.astype(np.int32)
    assert len(blocks) == len(assigs)
    print "kmeans avg bin size: {:.3f}".format(len(blocks) / float(k))
    print "kmeans bincounts: ", np.bincount(assigs)
    # all_rows = np.arange(len(assigs))
    # centroids[:] = 0 # TODO rm
    # centroids = np.random.randn(*centroids.shape) * np.var(blocks) # TODO rm
    return (blocks - centroids[assigs, :]).astype(np.int32)


# ------------------------------------------------ online kmeans

def _all_rotations_of(X):
    """returns a matrix X' whose rows include all rotations of X's rows"""
    # return np.array([np.roll(row, i) for row in X for i in range(X.shape[1])],
    #                 dtype=X.dtype)
    return np.vstack([np.roll(X, i, axis=1) for i in range(X.shape[1])])


def _nn_idx(X, q, dist_func=dist.dists_sq):
    idx, _ = dist.knn(X, q, k=1, dist_func=dist_func)
    return idx


def _sum_of_squares(x):
    return np.sum(x * x, axis=-1)


class KmeansCompressor(object):
    __slots__ = 'it k centroids shift_amt maxpool_phase counts optional'.split()

    # def __init__(self, block_sz=8, k=16, shift_amt=3, maxpool_phase=False):
    def __init__(self, block_sz=8, k=16, shift_amt=3, maxpool_phase=True, optional=True):
        self.k = k
        self.centroids = np.empty((k, block_sz), dtype=np.int32)
        self.centroids[0, :] = 0
        self.it = 0
        self.shift_amt = shift_amt
        self.maxpool_phase = maxpool_phase  # whether to compare to all rotations of centroid
        self.counts = np.zeros(k, dtype=np.int32)
        self.optional = optional

        print "KmeansCompressor: k={}, maxpool={}".format(k, maxpool_phase)

    def feed_block(self, block):
        assert len(block) == self.centroids.shape[1]
        assert len(block.shape) == 1  # TODO rm 1d constraint
        block = block.astype(np.int64)

        self.it += 1  # increment at top because centroid 0 is reserved

        block_sz = len(block)
        ncentriods = min(self.it, self.k)

        if self.it % 100 == 1:
            print "----- block num = {}".format(self.it - 1)

        if self.maxpool_phase:
            centroids = _all_rotations_of(self.centroids[:ncentriods, :])
            raw_idx = _nn_idx(centroids, block)
            # these steps are because _all_rotations_of() rolls the entire
            # matrix at once (as a likely premature optimization...), so
            # the rotated versions of a given centroid happen every ncentroids
            # rows in the expanded matrix
            idx = raw_idx % ncentriods
            rotation = raw_idx // ncentriods
            sub_block = centroids[raw_idx].ravel()
            # out_block = block.astype(np.int32) - centroids[raw_idx].ravel()
        else:
            centroids = self.centroids[:ncentriods, :]
            # print "self.centroids: ", self.centroids
            # print "ncentroids: ", ncentriods
            # print "centroids: ", centroids
            idx = _nn_idx(centroids, block)
            rotation = 0
            # out_block = block - centroids[idx].ravel()
            sub_block = centroids[idx].ravel()

        out_block = block - sub_block
        used_centroid = True
        # if self.optional:  # don't always sub nn centroid
        if self.optional and self.it > 8*self.k:  # not optional initially
            bits_orig = np.max(compress.nbits_cost(block))
            bits_sub = np.max(compress.nbits_cost(out_block))
            bitsave = bits_orig - bits_sub
            if bitsave < int(np.log2(self.k)):  # centroid doesn't help
                out_block = block
                used_centroid = False

        broken = not ((_sum_of_squares(block) > _sum_of_squares(out_block)) or idx == 0)
        broken = broken and used_centroid
        if broken:
            print "centroids:\n", centroids
            print "nn idx: ", idx
            print "block", block
            print "out_block", out_block
            print "block shape, out_block shape", block.shape, out_block.shape
            print "block var, out_block var, ", np.var(block), np.var(out_block)
            assert False

        # print "idx: ", idx
        # print "block shape, rotation", block.shape, rotation
        update_block = np.roll(block, -rotation, axis=0) if self.maxpool_phase else block

        # ------------------------ try different update strategies

        update_algo = 'seed_start'
        # update_algo = 'replace_rare_exact'

        if update_algo == 'seed_start':
            if ncentriods >= self.k:  # after first k-1 blocks
                if used_centroid:
                    self.counts[idx] += 1  # for debugging

                    if idx > 0:  # centroid 0 is always just 0s
                        # # moving avg of all blocks assigned to this centroid; this sets:
                        # #   alpha = 1 / (2^shift_amt)
                        # #   z = (1-alpha) * z + alpha * block
                        # delta = (update_block - centroids[idx]) >> self.shift_amt

                        # TODO rm after debug
                        # delta = (update_block - centroids[idx]) / float(self.counts[idx] + 1)
                        delta = (update_block - centroids[idx]) / self.counts[idx]

                        self.centroids[idx] += delta.astype(np.int64)
            else:
                self.centroids[ncentriods, :] = update_block
                self.counts[ncentriods] += 1

        if update_algo == 'replace_rare_exact':
            pass

            # SELF: pick up here

        return out_block, idx * block_sz + rotation if self.maxpool_phase else idx


def sub_online_kmeans(blocks, k=16, verbose=1, **kwargs):
    blocks = blocks.astype(np.int64)
    encoder = KmeansCompressor(k=k, block_sz=blocks.shape[1])
    out = np.empty(blocks.shape, dtype=np.int32)
    for i, block in enumerate(blocks):
        out[i], _ = encoder.feed_block(block)

    print "online kmeans centroid counts:\n", encoder.counts
    # import matplotlib.pyplot as plt
    # # plt.close()
    # plt.figure()
    # # plt.hist(encoder.counts)
    # plt.hist(encoder.counts, bins=np.arange(np.max(encoder.counts) + 1))
    # # plt.show()

    return out

    # centroids, assigs = dist.kmeans(blocks, k=k)
    # centroids = centroids.astype(np.int32)
    # assert len(blocks) == len(assigs)
    # # all_rows = np.arange(len(assigs))
    # # centroids[:] = 0 # TODO rm
    # # centroids = np.random.randn(*centroids.shape) * np.var(blocks) # TODO rm
    # return (blocks - centroids[assigs, :]).astype(np.int32)


# ================================================================ Vector AR

# def var_transform(blocks, ntaps=4, chunk_sz=-1):
def var_transform(blocks, ntaps=4, chunk_sz=-1):
    x = blocks.ravel()
    windows = window.sliding_window_1D(x, ntaps)
    y = windows[1:, -1]  # last col (which is next val), except from first row
    X = windows[:-1]  # all windows except last
    N = len(X)

    out = np.zeros_like(blocks).ravel()
    out[:ntaps] = x[:ntaps]

    if chunk_sz < 1:
        chunk_sz = N

    nchunks = int(np.ceil(N / float(chunk_sz)))
    for i in range(nchunks):
        start_idx = i * chunk_sz
        end_idx = min(N, start_idx + chunk_sz)

        # est = linear.LinearRegression(fit_intercept=True, normalize=False)
        est = linear.LinearRegression(fit_intercept=False, normalize=False)
        # est = linear.HuberRegressor(fit_intercept=False, alpha=0, max_iter=5)
        X_chunk = X[start_idx:end_idx]
        y_chunk = y[start_idx:end_idx]
        yhat = est.fit(X_chunk, y_chunk).predict(X_chunk)
        errs = y_chunk - yhat.astype(out.dtype)

        # print "start idx, end idx", start_idx, end_idx
        # print "errs.shape", errs.shape

        out[(start_idx + ntaps):(end_idx + ntaps)] = errs

        # print "linreg coeffs, intercept: {}, {:.3f}".format(est.coef_, est.intercept_)

    # print "blocks shape: ", blocks.shape
    # print "windows shape: ", windows.shape
    # print "X[:10], y[:10]\n", X[:10], y[:10]

    # print "X[:10], y[:10]", X[:10], y[:10]

    # out = np.zeros_like(blocks).ravel()
    # out[:ntaps] = x[:ntaps]

    # out[ntaps:] = errs
    return out.reshape(blocks.shape)


# ================================================================ main

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
