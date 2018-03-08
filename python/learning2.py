#!/usr/bin/env python

from __future__ import division

import collections
# import itertools
import numpy as np
# from sklearn import linear_model as linear  # for VAR

# from .utils import sliding_window as window
# from .utils.distance import kmeans, dists_sq
# from .utils import distance as dist

# from python import compress


# ================================================================ shifts lut

SHIFT_PAIRS_16 = [
    (7, 1),     # ~0    - .5    = ~-.5
    (3, 1),     # .125  - .5    = -.375
    (2, 1),     # .25   - .5    = -.25
    # (4, 2),     # .0625 - .25   = -.1875
    (3, 2),     # .125  - .5    = -.125
    (4, 3),     # .0625 - .125  = -.0625
    (0, 0),     # 1     - 1     = 0
    (3, 4),     # .125  - .0625 = .0625
    (2, 3),     # .25   - .125  - .125
    (2, 4),     # .25   - .0625 = .1875
    (1, 2),     # .5    - .25   = .25
    (1, 3),     # .5    - .125  = .375
    (0, 1),     # 1     - .5    = .5
    (0, 2),     # 1     - .25   = .75
    (0, 3),     # 1     - .125  = .875
    (0, 4),     # 1     - .0625 = .9375
    (0, 7),     # 1     - ~0    = ~1
    ]


# should be equivalent to `all_shifts(max_shift=5, omit_duplicates=True)`
# EDIT: wait, no, not true because we have shifts of 7 at the ends
SHIFT_PAIRS_26 = [
    (7, 1),     # ~0    - .5    = ~-.5
    (5, 1),     # .0625 - .5    = -.46875       # added
    (4, 1),     # .0625 - .5    = -.4375        # added, max 4
    (3, 1),     # .125  - .5    = -.375
    (2, 1),     # .25   - .5    = -.25
    (5, 2),     # .03125- .25   = -.21875
    (4, 2),     # .0625 - .25   = -.1875        # added, max 4
    (3, 2),     # .125  - .25   = -.125
    (5, 3),     # .03125- .125  = -.09375       # added
    (4, 3),     # .0625 - .125  = -.0625
    (5, 4),     # .03125- .0625 = -.03125       # added
    (0, 0),     # 1     - 1     = 0
    (4, 5),     # .0625 - .03125= .03125
    (3, 4),     # .125  - .0625 = .0625
    (3, 5),     # .125  - .03125= .09375        # added
    (2, 3),     # .25   - .125  - .125
    (2, 4),     # .25   - .0625 = .1875
    (2, 5),     # .25   - .03125= .21875        # added
    (1, 2),     # .5    - .25   = .25
    (1, 3),     # .5    - .125  = .375
    (1, 4),     # .5    - .0625 = .4375         # added, max 4
    (1, 5),     # .5    - .03125= .46875        # added
    (0, 1),     # 1     - .5    = .5
    (0, 2),     # 1     - .25   = .75
    (0, 3),     # 1     - .125  = .875
    (0, 4),     # 1     - .0625 = .9375
    (0, 5),     # 1     - .03125= .96875        # added
    (0, 7),     # 1     - ~0    = ~1
    ]


def all_shifts(max_shift=-1, omit_duplicates=True):
    vals = {}
    nbits = 8
    x = 1 << nbits  # reference val; 256 for nbits
    if max_shift < 0:
        max_shift = nbits - 1
    if omit_duplicates:
        vals[(0, 0)] = 0
    for a in range(max_shift + 1):
        for b in range(max_shift + 1):
            if omit_duplicates and a == b:
                continue
            vals[(a, b)] = (x >> a) - (x >> b)

    keys, coeffs = zip(*vals.items())
    keys = np.array(keys)
    coeffs = np.array(coeffs)
    order = np.argsort(coeffs)
    # print "shift results:"
    # print keys[order]
    # print coeffs[order]

    return keys[order], coeffs[order]


# okay, looks like (according to test immediately below) these values are
# identical to what's in our existing LUT; this makes sense given that impls
# are basically identical
def _i16_for_shifts(pos_shift, neg_shift, nbits=8):
    start_val = 1 << nbits  # 256 for nbits = 8
    return (start_val >> pos_shift) - (start_val >> neg_shift)


# TODO actual unit test
def _test_shift_coeffs(nbits=8):
    shifts, shift_coeffs = all_shifts()
    for (pos_shift, neg_shift), coeff in zip(shifts, shift_coeffs):
        assert _i16_for_shifts(pos_shift, neg_shift) == coeff

        for val in range(-128, 128):
            two_shifts_val = (val >> pos_shift) - (val >> neg_shift)
            # ya, this fails; multiply and rshift != using shifts directly
            # assert (val * coeff) >> nbits == two_shifts_val

            # this way works; requires two multiplies though...
            pos_coef = 1 << (nbits - pos_shift)
            neg_coef = 1 << (nbits - neg_shift)
            pos = (val * pos_coef) >> nbits
            neg = (val * neg_coef) >> nbits
            assert pos - neg == two_shifts_val

            # this way also fails
            # pos = val * pos_coef
            # neg = val * neg_coef
            # assert (pos - neg) >> nbits == two_shifts_val

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

    def __init__(self, block_sz=8, verbose=0, method='linreg',
                 shifts=SHIFTS, shift_coeffs=SHIFT_COEFFS, numbits=8, ntaps=1):
        # self.prev0 = 0
        # self.prev1 = 0
        # self.mod = 1 << nbits
        # self.shift0 = 0
        # self.shift1 = 1
        self.block_sz = block_sz
        self.verbose = verbose
        self.method = method
        self.shifts = shifts
        self.shift_coeffs = shift_coeffs
        self.numbits = numbits
        self.ntaps = ntaps

        self.last_val = 0
        self.last_delta = 0
        self.coef = 0
        self.coef = 256
        self.counter = 0
        # self.counter = 256 << (1 + self.numbits - 8)  # TODO indirect to learning rate, not just 1 # noqa
        # self.counter = 8 << 1  # equivalent to adding 8 to round to nearest?
        # self.counter = self.coef
        self.t = 0
        self.grad_counter = 0
        self.offset = 0
        self.offset_counter = 0

        shift_by = (1 + self.numbits - 8)
        self.coeffs = np.zeros(self.ntaps, dtype=np.int32) + 256
        self.counters = np.zeros(self.ntaps, dtype=np.int32) + (256 << shift_by)

        # self.approx_256_over_x = 1

        self.Sxy = 0
        self.Sxx = 0

        self.errs = []

        # print "using shifts, coeffs:"
        # print shifts
        # print shift_coeffs

        # for logging
        # self.best_idx_offset_counts = np.zeros(3, dtype=np.int64)
        self.best_idx_counts = np.zeros(len(self.shifts), dtype=np.int64)
        # counts_len = len(self.shifts) if method == 'linreg' else 512
        # self.best_idx_counts = np.zeros(counts_len, dtype=np.int64)
        self.best_coef_counts = collections.Counter()
        self.best_offset_counts = collections.Counter()

    def feed_group(self, group):
        pass  # TODO determine optimal filter here

        # errhat = a*x0 - b*x0 - a*x1 + b*x1
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

        x = diffs
        y = deltas

        # linear regression
        if self.method == 'linreg':

            Sxy = np.sum(x * y)
            Sxx = np.sum(x * x)
            # print "x, y dtypes: ", x.dtype, y.dtype
            # print "Sxx, Sxy dtypes: ", Sxx.dtype, Sxy.dtype
            coeff = (Sxy << 8) / Sxx  # shift to mirror what we'll need to do in C

            idx = binary_search(self.shift_coeffs, coeff)

            def compute_errs(x, y, shifts):
                predictions = (x >> shifts[0]) - (x >> shifts[1])
                return y - predictions

            # These are commented out because, empirically, they're
            # *never* chosen
            #
            # best_idx_offset = 0
            #
            # def compute_total_cost(errs, block_sz=self.block_sz):
            #     raw_costs = compress.nbits_cost(errs)
            #     block_costs_rows = raw_costs.reshape(-1, block_sz)
            #     block_costs = np.max(block_costs_rows, axis=1)
            #     return np.sum(block_costs)
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
            # self.best_idx_offset_counts[best_idx_offset] += 1

            errs = compute_errs(x, y, self.shifts[idx])

            self.best_idx_counts[idx] += 1  # for logging

        elif self.method == 'gradient':
            # update coeffs using last entry in each block

            # learning_rate_shift = 7  # learning rate of 2^(-learning_rate_shift)
            # learning_rate_shift = 8  # learning rate of 2^(-learning_rate_shift)
            # learning_rate_shift = 12  # learning rate of 2^(-learning_rate_shift)
            # learning_rate_shift = 4  # learning rate of 2^(-learning_rate_shift)
            # learning_rate_shift = 2  # learning rate of 2^(-learning_rate_shift)

            predictions = (x * self.coef) >> int(min(self.numbits, 8))
            for tap_idx in range(1, self.ntaps):
                predictions[tap_idx:] += (x[:-tap_idx] * self.coeffs[tap_idx])
            predictions += self.offset
            errs = y - predictions

            for b in range(8):  # for each block
                # only update based on a few values for efficiency
                which_idxs = 8 * b + np.array([3, 7])  # downsample by 4
                # which_idxs = 8 * b + np.array([1, 3, 5, 7])  # downsample by 2
                grads = 0
                # grads = np.zeros(self.ntaps)
                # offsets = 0
                for idx in which_idxs:
                    xval = x[idx]
                    # xval = x[idx] >> (self.numbits - 8)
                    # grad = int(-errs[idx] * x[idx]) >> 8
                    # grad = int(-errs[idx] * x[idx]) // 256
                    # y0 = np.abs(self.approx_256_over_x) * np.sign(xval)
                    # y0 = 1 + (256 - xval) >> 8
                    # y0 = 3 - ((3 * xval) >> 8)
                    # grad = int(-(errs[idx] << 8) / xval) if xval != 0 else 0  # works great
                    # self.counter -= grad  # equivalent to above two lines
                    # if self.t % 100 == 0:
                    #     print "grad:", grad
                    # continue

                    # # xabs = self.t # TODO rm
                    # xabs = np.abs(xval)
                    # if xabs == 0:
                    #     lzcnt = self.numbits
                    # else:
                    #     lzcnt = self.numbits - 1 - int(np.log2(xabs))
                    # lzcnt = max(0, lzcnt - 1)  # round up to nearest power of 2
                    # # lzcnt = min(15, lzcnt + 1)  # round up to nearest power of 2
                    # # numerator = 1 << self.numbits
                    # # recip = 1 << (lzcnt - 8) if lzcnt >= 8 else
                    # # recip = np.sign(xval) << (8 + lzcnt)
                    # shift_amt = max(0, lzcnt - (self.numbits - 8))  # usually 0, maybe 1 sometimes
                    # recip = (1 << shift_amt) * np.sign(xval)
                    # grad = int(-errs[idx] * recip)
                    # # grad = int(grad / len(which_idxs))

                    # normal grad descent
                    # grad = int(-errs[idx] * np.sign(xval))  # div by sqrt(hessian)
                    # grad = int(-errs[idx] * xval) >> self.numbits  # true gradient

                    # approx newton step for log(nbits)
                    err = errs[idx]
                    # if False:  # TODO rm
                    # if self.numbits > 8:
                    #     grad = int(-(1 + err)) if err > 0 else int(-(err - 1))
                    # else:
                    #     grad = int(-err)  # don't add 1
                    # self.grad_counter += (grad - (self.grad_counter >> 8))

                    # wtf this works so well for 16b, despite ignoring sign of x...
                    # (when also only shifting counter by learning rate, not
                    # an additional 8)
                    # grad = -err

                    # grad = -(err + np.sign(err)) * np.sign(xval)
                    # grad = -err * np.sign(xval)

                    # these both seem to work pretty well; prolly need to directly
                    # compare them
                    # grad = -err * np.sign(xval)
                    # grad = -np.sign(err) * xval  # significantly better than prev line
                    grad = np.sign(err) * xval  # significantly better than prev line
                    # ^ duh; above is minimizer for L1 loss
                    # grad = -np.sign(err) * np.sign(xval) << (self.numbits - 8)

                    # sub_from = ((1 << self.numbits) - 1) * np.sign(xval)
                    # approx_recip_x = sub_from - xval
                    # grad = -np.sign(err) * approx_recip_x

                    grads += int(grad)
                    # grads += grad >> 1  # does this help with overflow?

                    # simulate int8 overflow, adjusted for fact that we do 8 blocks
                    # per group (so 1024, 2048 instead of 128, 256)
                    mod = int(1 << self.numbits)
                    offset = mod // 2
                    grads = ((grads + offset) % mod) - offset
                    # grads = ((grads + 1024) % 2048) - 1024  # wrecks accuracy
                    # grads = ((grads + 8192) % 16384) - 8192  # no effect

                    self.errs.append(err)

                    # offsets += np.sign(err)  # optimize bias for l1 loss

                    # this is the other one we should actually consider doing
                    #
                    # grad = int(-errs[idx] * np.sign(xval))

                    # # approximation of what we'd end up doing with a LUT
                    # shift_to_just_4b = self.numbits - 4
                    # # y0 = ((xval >> shift_to_just_4b) + 1) << shift_to_just_4b
                    # shifted_xval = xval >> shift_to_just_4b
                    # if shifted_xval != 0:
                    #     y0 = int(256. / shifted_xval) << shift_to_just_4b
                    # else:
                    #     y0 = 16*np.sign(xval) << shift_to_just_4b
                    # # y0 = y0 * int(2 - (xval * y0 / 256))  # diverges
                    # y0 = int(256. / xval) if xval else 0

                    # y0 = (1 << int(8 - np.floor(np.log2(xval)))) * np.sign(xval)
                    # y0 = 4 * np.sign(xval)
                    # self.approx_256_over_x = int( y0*(2 - (int(xval*y0) >> 8)) ) # noqa # doesn't work
                    # grad = int(-errs[idx] * self.approx_256_over_x)
                    # grad = int(-errs[idx] * y0)
                    # grad = int(-errs[idx] * xval) # works
                    # grad = int(-errs[idx] * 2*np.sign(xval))

                    # this_best_coef = self.coef - grad
                    # self.counter += this_best_coef - self.coef
                    # self.counter -= grad  # equivalent to above two lines
                    # self.counter -= grad >> learning_rate_shift
                    # if self.t < 8:
                    # if self.t % 50 == 0:
                # if (self.t < 5 == 0) and (b == 0):
                # if (self.t % 50 == 0) and (b == 0):
                #     # print "errs: ", errs[-7], errs[-5], errs[-3], errs[-1]
                #     print "t, b = ", self.t, b
                #     print "errs: ", errs[-10:]
                #     print "xs:   ", x[-10:]
                #     # print "sum(|xs|)", np.sum(np.abs(x))
                #     print "grads: ", grads
                #     print "counter:", self.counter
                #     # print "grad counter:", self.grad_counter
                #     #     # print "recip, grad: ", recip, grad
                # self.coef = self.counter >> min(self.t, learning_rate_shift)
                # self.coef = self.counter >> learning_rate_shift

                learning_rate_shift = 1
                # learning_rate_shift = 4
                # grad_learning_shift = 1
                # grad_learning_shift = 4
                # offset_learning_shift = 4

                # compute average gradient for batch
                # grad = int(4 * grads / len(which_idxs))  # div by 16
                grad = int(grads / len(which_idxs))  # div by 64
                # grad = grads

                # self.grad_counter += grad - (self.grad_counter >> grad_learning_shift)
                # self.grad_counter += grad

                #
                # this is the pair of lines that we know works well for UCR
                #
                # self.counter -= grad
                self.counter += grad
                self.coef = self.counter >> (learning_rate_shift + (self.numbits - 8))

                # self.coef = self.counter >> learning_rate_shift
                # self.coef -= (self.grad_counter >> grad_learning_shift) >> learning_rate_shift
                # learn_shift = int(min(learning_rate_shift, np.log2(self.t + 1)))
                # self.coef = self.counter >> (learn_shift + (self.numbits - 8))
                # self.coef = self.counter >> learn_shift  # for use with l1 loss
                # self.coef -= (self.grad_counter >> grad_learning_shift) >> learn_shift
                # self.coef -= (self.grad_counter >> grad_learning_shift) >> learning_rate_shift
                # self.coef = 192  # global soln for olive oil

                # quantize coeff by rounding to nearest 16; this seems to help
                # quite a bit, at least for stuff that really should be double
                # delta coded (starlight curves, presumably timestamps)
                # self.coef = ((self.coef + 8) >> 4) << 4
                self.coef = (self.coef >> 4) << 4  # just round towards 0
                # self.coef = (self.coef >> 5) << 5  # just round towards 0

                # like above, but use sign since shift and unshift round towards 0
                # EDIT: no apparent difference, though perhaps cuz almost nothing
                # actually wants a negative coef
                # self.coef = ((self.coef + 8 * np.sign(self.coef)) >> 4) << 4

                # offset = int(offsets / len(which_idxs))  # div by 64
                # self.offset_counter += offset
                # # self.offset = self.offset_counter >> offset_learning_shift
                # self.offset = 0  # offset doesn't seem to help at all

                # self.coef = 0  # why are estimates biased? TODO rm
                # self.coef = 256

                # self.coef = self.counter
                # self.coef = np.clip(self.coef, -256, 256)  # apparently important
                # self.coef = np.clip(self.coef, -128, 256)  # apparently important

                # if self.t < 8:
                # if self.t % 100 == 0:
                #     print "----- t = {}".format(self.t)
                #     print "offset, offset counter: ", self.offset, self.offset_counter
                #     # print "grad, grads sum:   ", grad, grads
                #     # print "learn shift: ", learn_shift
                #     # print "errs[:10]:        ", errs[:16]
                #     # print "-grads[:10]: ", errs[:16] * x[:16]
                #     # print "signed errs[:10]: ", errs[:16] * np.sign(x[:16])
                #     print "new coeff, grad_counter, counter = ", self.coef, self.grad_counter, self.counter
                #     # print "new coeff, grad counter = ", self.coef, self.grad_counter

                # self.best_idx_counts[self.coef] += 1  # for logging
                self.best_coef_counts[self.coef] += 1
                self.best_offset_counts[self.offset] += 1

            # errs -= self.offset  # do this at the end to not mess up training

        elif self.method == 'exact':

            # print "using exact method"

            if self.numbits <= 8:
                predictions = (x * self.coef) >> self.numbits
            else:
                predictions = ((x >> 8) * self.coef)
            errs = y - predictions

            learn_shift = 6
            # shift = learn_shift + 2*self.numbits - 8
            shift = learn_shift

            # only update based on a few values for efficiency
            start_idx = 0 if self.t > 0 else 8
            for idx in np.arange(start_idx, len(x), 8):
                # xval = x[idx]  # >> (self.numbits - 8)
                # yval = y[idx]  # >> (self.numbits - 8)
                xval = x[idx] >> (self.numbits - 8)
                yval = y[idx] >> (self.numbits - 8)

                # # this way works just like global one, or maybe better
                # self.Sxx += xval * xval
                # self.Sxy += xval * yval

                # moving average way; seemingly works just as well
                # Exx = self.Sxx >> learn_shift
                # Exy = self.Sxy >> learn_shift
                Exy = self.Sxy >> shift
                Exx = self.Sxx >> shift
                # adjust_shift = 2 *
                diff_xx = (xval * xval) - Exx
                diff_xy = (xval * yval) - Exy
                self.Sxx += diff_xx
                self.Sxy += diff_xy

            # if min(self.Sxy, self.Sxx) >= 1024:
            #     self.Sxx /= 2
            #     self.Sxy /= 2

            Exy = self.Sxy >> shift
            Exx = self.Sxx >> shift
            self.coef = int((Exy << 8) / Exx)  # works really well

            # none of this really works

            # # print "Exy, Exx = ", Exy, Exx
            # print "xval, yval: ", xval, yval
            # print "diff_xx, diff_xy, Exy, Exx = ", diff_xx, diff_xy, Exy, Exx

            # # numerator = 1 << (2 * self.numbits)
            # numerator = 256
            # nbits = int(min(4, np.log2(Exx))) if Exx > 1 else 1
            # assert numerator >= np.abs(Exx)
            # # print "nbits: ", nbits
            # recip = int((numerator >> nbits) / (Exx >> nbits)) << nbits
            # # recip = recip >> (2 * self.numbits - 8)
            # print "numerator, recip: ", numerator, recip
            # self.coef = int(Exy * recip)

            self.best_coef_counts[self.coef] += 1

        self.t += 1
        return errs

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


def sub_online_regress(blocks, verbose=2, group_sz_blocks=8, max_shift=4,
                       only_16_shifts=True, method='linreg', numbits=8,
                       drop_first_half=False, **sink):
                       # drop_first_half=True, **sink):
    blocks = blocks.astype(np.int32)
    if only_16_shifts:
        shifts = SHIFT_PAIRS_16
        shift_coeffs = [_i16_for_shifts(*pair) for pair in shifts]
    else:
        shifts, shift_coeffs = all_shifts(max_shift=max_shift)
    encoder = OnlineRegressor(block_sz=blocks.shape[1], verbose=verbose,
                              shifts=shifts, shift_coeffs=shift_coeffs,
                              method=method, numbits=numbits)

    # print "using group_sz_blocks: ", group_sz_blocks
    # print "using method: ", method
    # print "using nbits: ", numbits

    out = np.empty(blocks.shape, dtype=np.int32)
    if group_sz_blocks < 1:
        group_sz_blocks = len(blocks)  # global model

    ngroups = int(len(blocks) / group_sz_blocks)
    for g in range(ngroups):
        # if verbose and (g > 0) and (g % 100 == 0):
        #     print "running on block ", g
        start_idx = g * group_sz_blocks
        end_idx = start_idx + group_sz_blocks
        group = blocks[start_idx:end_idx]
        errs = encoder.feed_group(group.ravel())
        out[start_idx:end_idx] = errs.reshape(group.shape)
    out[end_idx:] = blocks[end_idx:]

    if verbose > 1:
        if method == 'linreg':
            if group_sz_blocks != len(blocks):
                import hipsterplot as hp  # pip install hipsterplot
                # hp.plot(x_vals=encoder.shift_coeffs, y_vals=encoder.best_idx_counts,
                hp.plot(encoder.best_idx_counts,
                        num_x_chars=len(encoder.shift_coeffs), num_y_chars=12)
            else:
                coef_idx = np.argmax(encoder.best_idx_counts)
                coef = encoder.shift_coeffs[coef_idx]
                print "global linreg coeff: ", coef
        else:
            coeffs_counts = np.array(encoder.best_coef_counts.most_common())
            print "min, max coeff: {}, {}".format(
                coeffs_counts[:, 0].min(), coeffs_counts[:, 0].max())
            print "most common (coeff, counts):\n", coeffs_counts[:16]
            # bias_counts = np.array(encoder.best_offset_counts.most_common())
            # print "most common (bias, counts):\n", bias_counts[:16]

            errs = np.array(encoder.errs)
            print "raw err mean, median, std, >0 frac: {}, {}, {}, {}".format(
                errs.mean(), np.median(errs), errs.std(), np.mean(errs > 0))

    if drop_first_half and method == 'gradient':
        keep_idx = len(out) // 2
        out[:keep_idx] = out[keep_idx:(2*keep_idx)]
        print "NOTE: duplicating second half of data into first half!!" \
            " (blocks {}:)".format(keep_idx)

    return out


def _test_moving_avg(x0=0):
    # vals = np.zeros(5, dtype=np.int32) + 100
    vals = np.zeros(5, dtype=np.int32) - 100
    shft = 3
    counter = x0 << shft
    xhats = []
    for v in vals:
        xhat = counter >> shft
        xhats.append(xhat)
        counter += (v - xhat)

    print "vals:  ", vals
    print "xhats: ", xhats


# ================================================================ main

def main():
    np.set_printoptions(formatter={'float': lambda x: '{:.3f}'.format(x)})

    # print "all shifts:\n", all_shifts()
    # _test_shift_coeffs()

    _test_moving_avg()

    # print "shifts_16, coeffs"
    # print SHIFT_PAIRS_16
    # print [_i16_for_shifts(*pair) for pair in SHIFT_PAIRS_16]

    # x = np.array([5], dtype=np.int32)
    # print "shifting x left: ", x << 5

    # blocks = np.arange(8 * 64, dtype=np.int32).reshape(-1, 8)
    # sub_online_regress(blocks)


if __name__ == '__main__':
    main()
