#!/usr/bin/env python

from __future__ import division

# import itertools
import numpy as np
# from sklearn import linear_model as linear  # for VAR

# from .utils import sliding_window as window
# from .utils.distance import kmeans, dists_sq
# from .utils import distance as dist

from python import compress


# ================================================================ shifts lut

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
    # print "shift results:"
    # print keys[order]
    # print coeffs[order]

    return keys[order], coeffs[order]


# def coeff_lut():
#     """create lookup table `T` such that `T[coeff]` yields the two indices
#     whose associated coefficients are immediately above and below `coeff`"""
#     shifts, shift_coeffs = all_shifts()


SHIFTS, SHIFT_COEFFS = all_shifts()


# ================================================================ funcs

def binary_search(array, val):
    M = len(array)
    first = 0
    middle = int(M / 2)
    last = M - 1
    while (first <= last):
        middle_val = array[middle]
        if middle_val < val:
            first = middle + 1
        elif middle_val == val:
            return middle
        else:  # middle_val > val
            last = middle - 1
        middle = int((first + last) / 2)
    return middle


class OnlineRegressor(object):

    def __init__(self, block_sz=8, verbose=0):
        # self.prev0 = 0
        # self.prev1 = 0
        # self.mod = 1 << nbits
        # self.shift0 = 0
        # self.shift1 = 1
        self.block_sz = block_sz
        self.last_val = 0
        self.last_delta = 0
        self.verbose = verbose

        # for logging
        self.best_idx_offset_counts = np.zeros(3, dtype=np.int64)
        self.best_idx_counts = np.zeros(len(SHIFTS), dtype=np.int64)

    def feed_group(self, group):
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
        deltas = np.zeros(group.size, dtype=group.dtype)
        deltas[1:] = group[1:] - group[:-1]
        deltas[0] = group[0] - self.last_val
        self.last_val = group[-1]

        # deltas from previous time step; these are our indep variable
        diffs = np.zeros(group.size, dtype=group.dtype)
        diffs[1:] = deltas[:-1]
        diffs[0] = self.last_delta
        self.last_delta = deltas[-1]

        # linear regression
        x = diffs
        y = deltas
        Sxy = np.sum(x * y)
        Sxx = np.sum(x * x)
        # print "x, y dtypes: ", x.dtype, y.dtype
        # print "Sxx, Sxy dtypes: ", Sxx.dtype, Sxy.dtype
        coeff = (Sxy << 8) / Sxx  # shift to mirror what we'll need to do in C

        idx = binary_search(SHIFT_COEFFS, coeff)

        def compute_errs(x, y, shifts):
            predictions = (x >> shifts[0]) - (x >> shifts[1])
            return y - predictions

        def compute_total_cost(errs, block_sz=self.block_sz):
            raw_costs = compress.nbits_cost(errs)
            block_costs_rows = raw_costs.reshape(-1, block_sz)
            block_costs = np.max(block_costs_rows, axis=1)
            return np.sum(block_costs)

        best_idx_offset = 0
        errs = compute_errs(x, y, SHIFTS[idx])
        ret = errs

        # These are commented out because, empirically, they're *never* chosen
        #
        # cost = compute_total_cost(errs)
        # if idx > 0:
        #     errs2 = compute_errs(x, y, SHIFTS[idx - 1])
        #     cost2 = compute_total_cost(errs)
        #     if cost2 < cost:
        #         ret = errs2
        #         best_idx_offset = -1
        # if idx < (len(SHIFTS) - 1):
        #     errs3 = compute_errs(x, y, SHIFTS[idx + 1])
        #     cost3 = compute_total_cost(errs)
        #     if cost3 < cost:
        #         ret = errs3
        #         best_idx_offset = 1

        # for logging
        self.best_idx_offset_counts[best_idx_offset] += 1
        self.best_idx_counts[idx + best_idx_offset] += 1

        return ret

# while (first <= last) {
#    if (array[middle] < search)
#       first = middle + 1;
#    else if (array[middle] == search) {
#       printf("%d found at location %d.\n", search, middle+1);
#       break;
#    }
#    else
#       last = middle - 1;

#    middle = (first + last)/2;
# }

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


def sub_online_regress(blocks, verbose=2, group_sz_blocks=8):
    blocks = blocks.astype(np.int32)
    encoder = OnlineRegressor(block_sz=blocks.shape[1], verbose=verbose)
    out = np.empty(blocks.shape, dtype=np.int32)
    ngroups = int(len(blocks) / group_sz_blocks)
    for g in range(ngroups):
        if verbose and (g > 0) and (g % 100 == 0):
            print "running on block ", g
        start_idx = g * group_sz_blocks
        end_idx = start_idx + group_sz_blocks
        group = blocks[start_idx:end_idx]
        errs = encoder.feed_group(group.ravel())
        out[start_idx:end_idx] = errs.reshape(group.shape)
    out[end_idx:] = blocks[end_idx:]

    if verbose > 1:
        # former is whether we should use binary search result or index
        # immediately before or after it
        print " best idx offset counts: ", encoder.best_idx_offset_counts
        print " best idx counts: ", encoder.best_idx_counts

    return out


# ================================================================ main

def main():
    np.set_printoptions(formatter={'float': lambda x: '{:.3f}'.format(x)})

    # print "all shifts:\n", all_shifts()

    # x = np.array([5], dtype=np.int32)
    # print "shifting x left: ", x << 5

    blocks = np.arange(8 * 64, dtype=np.int32).reshape(-1, 8)
    sub_online_regress(blocks)


if __name__ == '__main__':
    main()
