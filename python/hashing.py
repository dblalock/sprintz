#!/usr/bin/env python

from __future__ import division
import numpy as np

# from .utils import sliding_window as window


class SignSubMeanHash(object):

    def __init__(self, block_sz, stride=1, shift=1, **sink):
        self.block_sz = block_sz
        self.stride = stride
        self.mean = 0
        self.last_pt = 0
        self.reduce_vect = np.array([1 << i for i in range(block_sz)])

        # print "reduct vect: ", self.reduce_vect

    def __call__(self, window):
        # ret = np.random.randint(256) # TODO real impl

        # update mean
        # delta = x - (self.mean >> self.shift)
        # window = np.atleast_1d(window)
        # self.mean += (window[-1] - self.last_pt)
        # self.last_pt = window[self.stride-1]

        self.mean = np.mean(window)

        mask = window < self.mean
        return np.dot(self.reduce_vect, mask)  # bool array to idx


class SignDeltaHash(object):

    def __init__(self, block_sz, stride=1, **sink):
        self.block_sz = block_sz
        self.stride = stride
        self.last_pt = 0
        self.reduce_vect = np.array([1 << i for i in range(block_sz - 1)])

    def __call__(self, window):
        mask = window[1:] < window[:-1]
        ret = np.dot(self.reduce_vect, mask)
        ret += (1 << (self.block_sz - 1)) * (window[0] < self.last_pt)
        self.last_pt = np.atleast_1d(window)[self.stride-1]
        return ret


class LeadingBitsHash(object):

    def __init__(self, block_sz, stride=1, **sink):
        self.block_sz = block_sz

    def __call__(self, window):
        return np.atleast_1d(window)[-1] >> 8  # XXX assumes 16b encoding


class BitsAndShapeHash(object):

    def __init__(self, block_sz, stride=1, **sink):
        pass

    def __call__(self, window):
        window = np.atleast_1d(window)
        keep_nbits = 3
        ret = window[-1] >> (16 - keep_nbits)  # XXX assumes 16b encoding
        ret += (window[0] <= window[-1]) << keep_nbits  # msb based on inc vs dec
        return ret


class BitsFirstLastHash(object):

    def __init__(self, block_sz, stride=1, **sink):
        pass

    def __call__(self, window):
        window = np.atleast_1d(window)
        keep_nbits = 3
        mask = (1 << keep_nbits) - 1
        # assert mask == 7
        ret = window[-1] >> (16 - 2 * keep_nbits)
        ret -= np.bitwise_and(ret, mask)  # subtract off low bits
        ret += window[0] >> (16 - keep_nbits)
        return ret


class DiffHash(object):
    def __init__(self, block_sz, stride=1, **sink):
        pass

    def __call__(self, window):
        window = np.atleast_1d(window)
        keep_nbits = 8
        diff = window[-1] - window[0]
        return diff >> (16 - keep_nbits)  # XXX assumes 16b encoding


class DiffAndPosHash(object):
    def __init__(self, block_sz, stride=1, **sink):
        pass

    def __call__(self, window):
        window = np.atleast_1d(window)
        keep_nbits_diff = 6
        keep_nbits_pos = 8 - keep_nbits_diff
        diff = window[-1] - window[0]
        ret = diff >> (16 - keep_nbits_diff)  # XXX assumes 16b encoding
        ret += (window[-1] >> (16 - keep_nbits_pos)) << keep_nbits_diff
        return ret


class Hashes:
    SIGN_SUB_MEAN = 'sign_sub_mean'
    SIGN_DELTA = 'sign_delta'
    LEADING_BITS = 'leading_bits'
    BITS_SHAPE = 'bits_shape'
    BITS_FIRST_LAST = 'BitsFirstLastHash'
    DIFF = 'DiffHash'
    DIFF_POS = 'DiffAndPosHash'

    _NAME_2_CTOR = {
        SIGN_SUB_MEAN: SignSubMeanHash,
        SIGN_DELTA: SignDeltaHash,
        LEADING_BITS: LeadingBitsHash,
        BITS_SHAPE: BitsAndShapeHash,
        BITS_FIRST_LAST: BitsFirstLastHash,
        DIFF: DiffHash,
        DIFF_POS: DiffAndPosHash,
    }


class HashCompressor(object):

    # def __init__(self, initial_block, hash_name, stride=1, tbl_size=256):
    # def __init__(self, hash_name, stride=1, shift=2, tbl_size=256):
    def __init__(self, hash_name, block_sz=8, stride=1, shift=2, tbl_size=256,
                 # can_choose=False, **sink):
                 can_choose=True, **sink):
        self.hash_name = hash_name
        self.block_sz = block_sz
        self.stride = stride  # length of vects stored in hash table
        self.shift = shift  # for moving avg used to update vals in table
        self.can_choose = can_choose  # can choose to not use value in table?
        self.tbl_size = tbl_size
        self.tbl = np.zeros((tbl_size, stride), dtype=np.int32)

        self.hash = Hashes._NAME_2_CTOR[hash_name](
            block_sz=block_sz, stride=stride, shift=shift, tbl_size=tbl_size)

        # self.iter = 0 # TODO rm
        self.counts = np.zeros(tbl_size, dtype=np.int32)  # for data viz

        # create hasher object
        # self.hash, f_init = Hashes._NAME_2_CTOR[hash_name]
        # self.hash_state = f_init(stride=stride, shift=shift, tbl_size=tbl_size)

    def feed_window(self, x, y):
        """x determines hash bucket, and y is what we encode; latter
        defaults to x if not supplied"""
        # idx, self.hash_state = self.hash(x, self.hash_state)

        # y = x[-self.stride:] if (y is None) else y

        # print self.iter, ") x shape, y shape", x.shape, y.shape

        # return y  # TODO rm

        idx = self.hash(x)
        # TODO RM and uncomment above
        # idx = 0

        self.counts[idx] += 1

        err = y - (self.tbl[idx] >> self.shift)

        if self.can_choose and np.max(np.abs(err)) > np.max(np.abs(y)):
            return y  # using hash table wouldn't help

        self.tbl[idx] += err
        return err


def _first_derivs(x):
    out = np.zeros(x.size, dtype=x.dtype)
    out[0] = x.ravel()[0]
    out[1:] = np.diff(x.ravel())
    return out.reshape(x.shape)


# default_hash = Hashes.SIGN_SUB_MEAN
# default_hash = Hashes.SIGN_DELTA
# default_hash = Hashes.LEADING_BITS
# default_hash = Hashes.BITS_SHAPE
# default_hash = Hashes.BITS_FIRST_LAST
# default_hash = Hashes.DIFF
default_hash = Hashes.DIFF_POS


def hash_predict_transform(blocks, hash_name=default_hash,
                           stride=1, predict_deltas=True, **kwargs):

    block_sz = blocks.shape[1]
    x = blocks.ravel().astype(np.int64)

    # x = _first_derivs(x) # TODO rm if not better

    encoder = HashCompressor(hash_name=hash_name, stride=stride, **kwargs)
    out = np.empty(x.size, dtype=np.int32)

    for start_idx in range(0, len(x) - block_sz, stride):
        end_idx = start_idx + block_sz
        predict_idx = end_idx  # since end index not inclusive
        y = x[predict_idx]
        # if True:
        # if False:
        if predict_deltas:
            y -= x[predict_idx - 1]

        # out[predict_idx] = y # TODO rm
        # continue

        context = x[start_idx:end_idx]
        out[predict_idx] = encoder.feed_window(x=context, y=y)

    out[:block_sz] = x[:block_sz]
    remainder = len(x) % stride
    if remainder > 0:
        out[-remainder:] = x[-remainder:]

    print "hash table counts:\n", encoder.counts
    # import matplotlib.pyplot as plt
    # # plt.close()
    # plt.figure()
    # # plt.hist(encoder.counts)
    # plt.hist(encoder.counts, bins=np.arange(np.max(encoder.counts) + 1))
    # plt.show()

    return out.reshape(blocks.shape)


# ================================================================ main

def main():
    pass


if __name__ == '__main__':
    main()
