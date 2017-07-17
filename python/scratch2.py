
import collections
# import itertools
# import os
# import matplotlib as mpl
# import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sb

# from .scratch1 import nbits_cost

from .compress import nbits_cost

# from .datasets import ucr
# from . import datasets as ds
# from .utils import files
# from .utils import sliding_window as window


# class SprintzEncoder(object):

#     def __init__(self, history_len, nn_step, hash_algo='perm'):
#         self.nn_step = nn_step
#         self.hash_algo = hash_algo
#         # self.history = np.zeros(history_len
#         self.history_len = history_len  # TODO have a circ buff

#     def encode(self, all_samples):  # TODO enc one block at a time
#         """all_samples: 1D array"""
#         diffs =


# def _pseudocounts(nbits):
#     assert nbits == 8

#     vals = np.arange(-128, 128)
#     costs = nbits_cost(vals)
#     pseudocounts = nbits - costs

#     assert np.min(pseudocounts) == 0
#     return pseudocounts


# Symbol = collections.namedtuple('Symbol', 'val count'.split())

class Symbol(object):
    __slots__ = 'val count'.split()

    def __init__(self, val, count):
        self.val = val
        self.count = count


def min_max_vals_for_nbits(nbits, signed=True):
    if signed:
        minval = -(1 << (nbits - 1))  # for 8b, -128
        maxval = -minval - 1  # for 8b, 127
        return minval, maxval
    return 0, (1 << nbits) - 1


def _initial_symbols_positions(nbits, signed=True):
    minval, maxval = min_max_vals_for_nbits(nbits, signed=signed)

    if signed:
        # vals = 0, -1, 1, -2, 2, etc
        vals = [0]
        for i in range(1, maxval + 1):
            vals += [-i, i]
        vals += [minval]
        vals = np.array(vals)

        idxs = np.array([int(np.where(vals == i)[0]) for i in np.arange(minval, maxval + 1)])
    else:
        # vals = 0, 1, 2, 3, etc
        idxs = np.arange(minval, maxval + 1)  # 0, 2^nbits - 1
        vals = np.copy(idxs)

    # pseudocount of nbits - (nbits needed to express the value)
    pseudocounts = [nbits - nbits_cost(val, signed=signed) for val in vals]
    symbols = [Symbol(val=val, count=count) for val, count in zip(vals, pseudocounts)]

    # print "using vals: ", vals
    # print "using idxs: ", idxs
    # ordered_vals = np.arange(minval, maxval + 1)
    # for i, val in enumerate(ordered_vals):
    #     idx = idxs[val + minval]
    #     assert vals[idx] == val

    return symbols, idxs.astype(np.int32)


class SortTransformer(object):
    __slots__ = 'symbols positions minval maxval'.split()

    def __init__(self, nbits=8, signed=True):
        self.symbols, self.positions = _initial_symbols_positions(nbits, signed=signed)
        # self.minval = -(1 << (nbits - 1))  # for 8b, -128
        self.minval, self.maxval = min_max_vals_for_nbits(nbits, signed=signed)

        # check that everything is sorted # TODO rm
        counts = self.counts()
        # print "initial counts: ", counts
        # print "minval, positions", self.minval, self.positions
        diffs = np.diff(counts)
        assert np.all(diffs <= 0) or False

    # we maintain two lists acting as maps:
    #   symbols: idx -> val, count
    #   positions: val -> idx
    def feed_val(self, val, move_limit=2):
        symbols = self.symbols  # abbreviate since written a lot here
        # idx = self.positions[val + self.minval]
        idx = self.positions[val - self.minval]
        symbol = Symbol(val=symbols[idx].val, count=symbols[idx].count)
        symbol.count += 1

        # assert self.minval <= val <= self.maxval

        # return # TODO rm

        # assert self.minval == 0

        it = 0
        move_limit = move_limit if move_limit > 0 else len(self.positions)
        # wow; making this >= instead of > helps an *enormous* amount
        # EDIT: that was before we were decrementing idx, so who knows what
        # that was doing...
        while symbol.count >= symbols[idx - 1].count and idx > 0 and it < move_limit:
            other_val = symbols[idx - 1].val

            # assert 0 <= other_val < 256
            # assert 0 <= val < 256

            # self.positions[other_val + self.minval] += 1
            # self.positions[val + self.minval] -= 1
            self.positions[other_val - self.minval] += 1
            self.positions[val - self.minval] -= 1

            # swap the two symbols; have to set the fields explicitly
            # for subtle python reasons; ie, obvious way below doesn't work:
            #
            # symbols[idx], symbols[idx - 1] = symbols[idx - 1], symbol
            #
            symbols[idx].val = symbols[idx - 1].val
            symbols[idx].count = symbols[idx - 1].count
            symbols[idx - 1].val = symbol.val
            symbols[idx - 1].count = symbol.count

            idx -= 1
            it += 1

        # # check that everything is sorted # TODO rm
        # counts = self.counts()
        # # print "updated vals: ", self.sorted_vals()
        # # print "updated positions: ", self.positions
        # # print "updated counts: ", counts
        # diffs = np.diff(counts)
        # assert np.all(diffs <= 0)

    def position_of_val(self, val):
        # return self.positions[val + self.minval]
        return self.positions[val - self.minval]

    def counts(self):
        return np.array([s.count for s in self.symbols])

    def sorted_vals(self):
        return np.array([s.val for s in self.symbols])


def sort_transform(blocks, nbits, prefix_nbits=-1,
                   # zigzag=True, sort_remainders=True):
                   zigzag=False,
                   # zigzag=True,
                   # sort_remainders=True):
                   sort_remainders=False):
    """
    >>> x = [3, -4, 0, 1, -2]
    >>> sort_transform(x, nbits=4)
    array([6, 7, 0, 2, 3])
    """

    # prefix_nbits = 9  # TODO rm
    # return (np.abs(blocks) << 1) - (blocks > 0).astype(np.int32) # TODO rm

    if sort_remainders:
        assert 0 < prefix_nbits < nbits  # no remainders without suffix bits
    if prefix_nbits is None or prefix_nbits < 1:
        prefix_nbits = nbits

    assert prefix_nbits <= nbits
    suffix_nbits = nbits - prefix_nbits
    max_trailing_val = max((1 << suffix_nbits) - 1, 0)

    blocks = np.asarray(blocks)
    out = np.empty(blocks.size, dtype=blocks.dtype)

    if zigzag:
        blocks = (np.abs(blocks) << 1) - (blocks > 0).astype(np.int32)

    signed = not zigzag
    # signed = True # TODO rm
    transformer_high = SortTransformer(nbits=prefix_nbits, signed=signed)
    if suffix_nbits and sort_remainders:
        transformer_low = SortTransformer(nbits=suffix_nbits, signed=signed)

    for i, val in enumerate(blocks.ravel()):
        val_low = np.bitwise_and(val, max_trailing_val)
        val_high = val >> suffix_nbits
        try:
            if suffix_nbits and sort_remainders:
                pos_high = transformer_high.position_of_val(val_high)
                pos_low = transformer_low.position_of_val(val_low)

                out[i] = (pos_high << suffix_nbits) + pos_low

                transformer_high.feed_val(val_high)
                transformer_low.feed_val(val_low)
            else:
                pos_high = transformer_high.position_of_val(val_high)
                out[i] = (pos_high << suffix_nbits) + val_low
                transformer_high.feed_val(val_high)

        except IndexError:  # happens when deltas overflow
            out[i] = (1 << (nbits - 1)) - 1

    return out.reshape(blocks.shape)


if __name__ == '__main__':
    x = [3, -4, 0, 1, -2]
    sort_transform(x, nbits=4)

    x = [3, -4, 0, 1, -2]
    sort_transform(x, nbits=4)

    # import doctest
    # doctest.testmod()
