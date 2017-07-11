#!/usr/bin/env python

# import itertools
import numpy as np

from scratch1 import my_old_transform, my_transform_inverse


def test_my_old_transform_nonincreasing_abs():
    x = np.zeros((1, 2), dtype=np.int32)

    # idxs = np.arange(256)
    # left_col = np.tile(idxs, 256).reshape((256, 256)).T.reshape((-1))
    # idx_pairs = np.array(itertools.product(idxs, idxs))

    # for i in xrange(256):
    for i in xrange(-128, 128):
        abs_i = np.abs(i)
        x[0, 0] = i
        # for j in xrange(256):
        for j in xrange(-128, 128):
            orig_abs = max(abs_i, np.abs(j))
            x[0, 1] = j

            x_enc = my_old_transform(x)
            new_abs = np.max(np.abs(x_enc))

            assert new_abs <= orig_abs


def test_my_transform_inverse():
    print "running inverse transform test"
    x = np.zeros((1, 2), dtype=np.int32)
    min_val = -128
    # min_val = 0
    max_val = 127

    for i in xrange(min_val, max_val + 1):
        x[0, 0] = i
        for j in xrange(min_val, max_val + 1):
            x[0, 1] = j
            x_enc = my_linear_transform(x)
            x_hat = my_transform_inverse(x_enc)

            eq = np.array_equal(x, x_hat)
            if not eq:
                print "failing x, x_hat", x, x_hat
                assert eq


def encode_decode(a, b):
    # ya, this one is just not a bijection; we lose b's LSB

    print "------------------------"

    beta = (a >> 1) + (b >> 1)
    alpha = a - beta

    ahat = alpha + beta
    tmp = (beta - ahat >> 1)
    bhat = tmp << 1
    print "tmp, orig bhat", tmp, bhat
    bhat -= (bhat >> 1) > tmp

    print "a, b: ", a, b
    print "alpha, beta: ", alpha, beta
    print "ahat, bhat: ", ahat, bhat

    assert a == ahat
    assert b == bhat


def encode_decode2(a, b):
    print "------------------------"

    beta = b - (a >> 1)
    alpha = a - (beta >> 1)
    # beta = (a >> 1) + (b >> 1)
    # alpha = a - beta

    ahat = alpha + (beta >> 1)
    bhat = beta + (a >> 1)

    print "a, b: ", a, b
    print "alpha, beta: ", alpha, beta
    print "ahat, bhat: ", ahat, bhat

    assert a == ahat
    assert b == bhat


def main():
    # a = 4
    # b = 6
    encode_decode2(4, 6)
    encode_decode2(6, 2)


if __name__ == '__main__':
    main()
