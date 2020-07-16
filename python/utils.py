#!/usr/bin/env python

import itertools

from joblib import Memory
import numpy as np
import pandas as pd
from sklearn import cluster
from scipy import signal

_memory = Memory('.', verbose=0)


# ================================================================ pandas

# XXX TODO support complea numbers (in particular, handle nans)
def allclose(a, b, rtol=1e-5, atol=1e-5, equal_nan=True):
    """Like numpb allclose, but handles pandas nullable scalar types"""
    if isinstance(a, (tuple, list)):
        a = np.array(a)
    if isinstance(b, (tuple, list)):
        b = np.array(b)

    # shape checks
    assert len(a) == len(b)
    assert a.shape == b.shape
    assert a.dtype == b.dtype

    # compare locations / presence of nans
    a_mask = pd.notna(a)
    b_mask = pd.notna(b)
    assert np.array_equal(a_mask, b_mask)
    if not equal_nan:
        assert np.all(a_mask)

    # extract and compare values at non-nan indices
    try:
        a_nonnan = a.iloc[a_mask]
    except (AttributeError, NotImplementedError):
        a_nonnan = a[a_mask]
    try:
        b_nonnan = b.iloc[b_mask]
    except (AttributeError, NotImplementedError):
        b_nonnan = b[b_mask]
    absdiffs = np.abs(a_nonnan - b_nonnan)
    return np.all(absdiffs <= (atol + rtol * np.abs(b_nonnan)))


def array_equal(a, b, equal_nan=True):
    return allclose(a, b, rtol=0, atol=0, equal_nan=equal_nan)


# ================================================================ misc

def ndecimal_digits(string):
    dec_idx = string.find('.') % len(string)  # -1 and last char same
    return len(string) - dec_idx - 1


def is_dict(x):
    return isinstance(x, dict)


def is_list_or_tuple(x):
    return isinstance(x, (list, tuple))


def as_list_or_tuple(x):
    return x if is_list_or_tuple(x) else [x]


def is_scalar_seq(x):
    try:
        [float(element) for element in x]
        return True
    except TypeError:
        return False


def as_scalar_seq(x):
    if is_scalar_seq(x):
        return x
    try:
        _ = float(x)
        return [x]
    except TypeError:
        raise TypeError("Couldn't convert value '{}' to sequence "
                        "of scalars".format(x))


def is_string(x):
    return isinstance(x, (str,))


def flatten_list_of_lists(l):
    return list(itertools.chain.from_iterable(l))


def element_size_bytes(x):
    return np.dtype(x.dtype).itemsize


def invert_permutation(permutation):
    return np.arange(len(permutation))[np.argsort(permutation)]


# ================================================================ image

def conv2d(img, filt, pad='same'):
    # assert pad in ('same',)  # TODO support valid
    # mode = 'constant'
    if len(img.shape) == 2:
        return signal.correlate2d(img, filt, mode=pad)

    # img is more than 2d; do a 2d conv for each channel and sum results
    assert len(img.shape) == 3
    out = np.zeros(img.shape[:2], dtype=np.float32)
    for c in range(img.shape[2]):
        f = filt[:, :, c] if len(filt.shape) == 3 else filt
        out += signal.correlate2d(img[:, :, c], f, mode=pad)
    return out


# def filter_img(img, filt):
#     out = conv2d(img, filt)
#     return out / np.max(out)


# ================================================================ distance

def dists_sq(X, q):
    diffs = X - q
    return np.sum(diffs * diffs, axis=-1)


def dists_l1(X, q):
    diffs = np.abs(X - q)
    return np.sum(diffs, axis=-1)


def sq_dists_to_vectors(X, queries, rowNorms=None, queryNorms=None):
    Q = queries.shape[0]

    mat_size = X.shape[0] * Q
    mat_size_bytes = element_size_bytes(X[0] + queries[0])
    if mat_size_bytes > int(1e9):
        print("WARNING: sq_dists_to_vectors: attempting to create a matrix" \
              "of size {} ({}B)".format(mat_size, mat_size_bytes))

    if rowNorms is None:
        rowNorms = np.sum(X * X, axis=1, keepdims=True)

    if queryNorms is None:
        queryNorms = np.sum(queries * queries, axis=1)

    dotProds = np.dot(X, queries.T)
    return (-2 * dotProds) + rowNorms + queryNorms  # len(X) x len(queries)


def all_eq(x, y):
    if len(x) != len(y):
        return False
    if len(x) == 0:
        return True
    return np.max(np.abs(x - y)) < .001


def top_k_idxs(elements, k, smaller_better=True, axis=-1):
    if smaller_better:  # return indices of lowest elements
        which_nn = np.arange(k)
        return np.argpartition(elements, kth=which_nn, axis=axis)[:k]
    else:  # return indices of highest elements
        which_nn = len(elements) - 1 - np.arange(k)
        return np.argpartition(elements, kth=which_nn, axis=axis)[-k:][::-1]


def compute_true_knn(X, Q, k=1000, print_every=5, block_sz=128):
    nqueries = Q.shape[0]
    nblocks = int(np.ceil(nqueries / float(block_sz)))

    truth = np.full((nqueries, k), -999, dtype=np.int32)

    if nqueries <= block_sz:
        dists = sq_dists_to_vectors(Q, X)
        assert dists.shape == (Q.shape[0], X.shape[0])
        for i in range(nqueries):
            truth[i, :] = top_k_idxs(dists[i, :], k)
            # truth[i, :] = top_k_idxs(dists[:, i], k)
        return truth

    for b in range(nblocks):
        # recurse to fill in knn for each block
        start = b * block_sz
        end = min(start + block_sz, nqueries)
        rows = Q[start:end, :]
        truth[start:end, :] = compute_true_knn(X, rows, k=k, block_sz=block_sz)

        if b % print_every == 0:
            print("computing top k for query block "
                  "{} (queries {}-{})...".format(b, start, end))

    # for i in range(nqueries):
    #     if i % print_every == 0:
    #         print "computing top k for query {}...".format(i)
    #     truth[i, :] = top_k_idxs(dists[i, :], k)
    print("done")

    assert np.all(truth != -999)
    return truth


def knn(X, q, k, dist_func=dists_sq):
    dists = dist_func(X, q)
    idxs = top_k_idxs(dists, k)
    return idxs, dists[idxs]


def orthonormalize_rows(A):
    Q, R = np.linalg.qr(A.T)
    return Q.T


def random_rotation(D):
    rows = np.random.randn(D, D)
    return orthonormalize_rows(rows)


def hamming_dist(v1, v2):
    return np.count_nonzero(v1 != v2)


def hamming_dists(X, q):
    return np.array([hamming_dist(row, q) for row in X])


if __name__ == '__main__':

    a = np.random.randn(10)
    sort_idxs = np.argsort(a)[::-1]
    print(a)
    print(top_k_idxs(a, 3, smaller_better=False))
    print(sort_idxs[:3])
