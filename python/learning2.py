#!/usr/bin/env python

from __future__ import division

# import itertools
import numpy as np
# from sklearn import linear_model as linear  # for VAR

# from .utils import sliding_window as window
# from .utils.distance import kmeans, dists_sq
# from .utils import distance as dist
# from . import compress


class OnlineRegressor(object):

    def __init__(self, nbits=8):
        self.prev0 = 0
        self.prev1 = 0
        self.mod = 1 << nbits
        self.shift0 = 0
        self.shift1 = 1

        self.last_val = 0
        self.last_delta = 0

    def feed_block(self, block):
        pass  # TODO determine optimal filter here

        # errhat = a*x0 - b*x0 - a*x1 - b*x1
        #   = a(x0 - x1) + b(x1 - x0)
        #   = c(x0 - x1), where c = (a - b)
        #
        # we should compute c, and find shifts (which correspond to a, b) that
        # approximate it well; also note that errhat is prediction of the delta
        #
        # this is just linear regression between (x0 - x1) and new val, with
        # some extra logic at the end to get shifts based on regression coeff

        # deltas; these are our target variable
        deltas = np.zeros(block.size)
        deltas[1:] = block[1:] - block[:-1]
        deltas[0] = block[0] - self.last_val
        self.last_val = block[-1]

        # deltas from previous time step; these are our indep variable
        diffs = np.zeros(block.size)
        self.diffs[1:] = deltas[:-1]
        diffs[0] = self.last_delta
        self.last_delta = deltas[-1]

        # linear regression (without multiplies or divides)
        # or maybe just find best (a, b) directly (only 8x8 = 64 options)

        # TODO

    def feed_val(self, val):
        delta = (val - self.prev0) % self.mod  # always delta code
        # diff = self.prev0 - self.prev1  # TODO just delta from prev time step
        prediction = (self.diff >> self.shift0) - (self.diff >> self.shift1)
        # add0 = (self.prev0 >> self.shift0) - (self.prev0 >> self.shift1)
        # add1 = (self.prev1 >> self.shift1) - (self.prev1 >> self.shift0)
        self.diff = delta
        self.prev1 = self.prev0
        self.prev0 = val
        return delta - prediction


def all_shifts():
    vals = {}
    nbits = 8
    x = 1 << nbits  # reference val
    for a in range(nbits):
        for b in range(nbits):
            vals[(a, b)] = (x >> a) - (x >> b)

    keys, coeffs = zip(*vals.items())
    keys = np.array(keys)
    coeffs = np.array(coeffs)
    order = np.argsort(coeffs)
    print "shift results:"
    print keys[order]
    print coeffs[order]


# ================================================================ main

def main():
    np.set_printoptions(formatter={'float': lambda x: '{:.3f}'.format(x)})

    # print "done"
    all_shifts()


if __name__ == '__main__':
    main()
