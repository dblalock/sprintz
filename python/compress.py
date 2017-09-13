#!/usr/bin/env python

import numpy as np


# ================================================================ Funcs

def nbits_cost(diffs, signed=True):
    """
    >>> [nbits_cost(i) for i in [0, 1, 2, 3, 4, 5, 7, 8, 9]]
    [0, 2, 3, 3, 4, 4, 4, 5, 5]
    >>> [nbits_cost(i) for i in [-1, -2, -3, -4, -5, -7, -8, -9]]
    [1, 2, 3, 3, 4, 4, 4, 5]
    >>> nbits_cost([])
    array([], dtype=int32)
    >>> nbits_cost([0, 2, 1, 0])
    array([0, 3, 2, 0], dtype=int32)
    >>> nbits_cost([0, 2, 1, 3, 4, 0], signed=False)
    array([0, 2, 1, 2, 3, 0], dtype=int32)
    """
    if diffs is None:
        return None

    diffs = np.asarray(diffs)
    if diffs.size == 0:
        return np.array([], dtype=np.int32)

    if not signed:
        assert np.all(diffs >= 0)
        pos_idxs = diffs > 0
        nbits = np.zeros(diffs.shape, dtype=np.int32)
        nbits[pos_idxs] = np.floor(np.log2(diffs[pos_idxs])) + 1
        nbits[~pos_idxs] = 0
        return nbits

    shape = diffs.shape
    diffs = diffs.ravel()
    equiv_diffs = np.abs(diffs) + (diffs >= 0).astype(np.int32)  # +1 if < 0
    nbits = np.ceil(np.log2(equiv_diffs)) + 1
    nbits = np.asarray(nbits, dtype=np.int32)  # next line can't handle scalar
    nbits[diffs == 0] = 0

    return nbits.reshape(shape) if nbits.size > 1 else nbits[0]  # unpack if scalar


def zigzag_encode(x):
    """
    >>> [zigzag_encode(i) for i in [0,1,-1,2,-2,3,-3]]
    [0, 1, 2, 3, 4, 5, 6]
    >>> zigzag_encode([0,1,-1,2,-2,3,-3])
    array([0, 1, 2, 3, 4, 5, 6], dtype=int32)
    """
    x = np.asarray(x, dtype=np.int32)
    return (np.abs(x) << 1) - (x > 0).astype(np.int32)


def zigzag_decode(x):
    return np.bitwise_xor(x >> 1, -np.bitwise_and(x, 1))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
