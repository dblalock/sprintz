#!/usr/bin/env python

import numpy as np


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
    >>> nbits_cost([0, 2, 1, 0], signed=False)
    array([0, 2, 1, 0], dtype=int32)
    """
    if diffs is None:
        return None

    diffs = np.asarray(diffs)
    if diffs.size == 0:
        return np.array([], dtype=np.int32)

    shape = diffs.shape
    diffs = diffs.ravel()
    equiv_diffs = np.abs(diffs) + (diffs >= 0).astype(np.int32)  # +1 if < 0
    nbits = np.ceil(np.log2(equiv_diffs)) + 1
    nbits = np.asarray(nbits, dtype=np.int32)  # next line can't handle scalar
    nbits[diffs == 0] = 0

    ret = nbits.reshape(shape) if nbits.size > 1 else nbits[0]  # unpack if scalar
    return ret if signed else np.maximum(0, ret - 1)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
