#!/usr/bin/env python

# generative model associated with compression approaches to give us insights
# into how to compress better

import numpy as np
import matplotlib.pyplot as plt

# from .datasets import viz


# ================================================================ main

def unif_nbits(N=200, B=16, M=8, deltaB=2):
    assert N % M == 0  # must use even number of blocks
    N_blocks = N / M

    # unif rand walk for deltas of nbits to encode each block
    #
    # min_delta = -(1 << (deltaB - 1)) - 1  # symmetric, so, eg, [-3, 3]
    # max_delta = -min_delta
    # print "min delta, max delta: ", min_delta, max_delta
    # delta_maxes = np.random.randint(min_delta, max_delta + 1,
    #                                 size=N_blocks-1, dtype=np.int32)

    # gauss rand walk for deltas of nbits to encode each block
    #
    # delta_maxes = np.random.randn(N_blocks - 1)
    # delta_maxes = np.floor(delta_maxes + .5).astype(np.int32)

    # compute maxes from above deltas (using either strategy)
    #
    # maxes = np.zeros(N_blocks, dtype=np.int32)
    # maxes[0] = B - 6
    # for i, el in enumerate(delta_maxes):
    #     maxes[i + 1] = np.clip(maxes[i] + delta_maxes[i], 0, B)

    # # print "maxes: ", maxes[:20]

    # gaussian max values instead of rand walk
    maxes = (np.random.randn(N_blocks) * B / 8) + B - 6
    maxes = np.clip(maxes, 0, B)
    maxes = np.floor(maxes + .5).astype(np.int32)

    dx = np.zeros((N_blocks, M), dtype=np.int32)
    for i, maxbits in enumerate(maxes):
        if maxbits == 0:
            dx[i, :] = 0
            continue
        minval = -(1 << (maxbits - 1))
        maxval = -minval - 1
        # if minval >= maxval:
        #     print "minval, maxval", minval, maxval
        dx[i, :] = np.random.randint(minval, maxval + 1, size=M)
    dx = dx.ravel()[1:]

    # print "dx: ", dx[:20]

    x = np.zeros(N, dtype=np.int32)
    minval = -(1 << (B - 1))
    maxval = -minval - 1
    x[0] = dx[0]
    for i, delta in enumerate(dx):
        x[i + 1] = np.clip(x[i] + delta, minval, maxval)
    return x

    # x = np.cumsum(dx)
    # minval = -(1 << (B - 1))
    # maxval = -B - 1
    # return np.clip(x, minval, maxval)


def main():
    B = 16  # number of bits
    N = 200  # number of samples
    M = 8  # block size

    niters = 5

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    # for i in range(2):
    # for i in range(10):

    axes[0, 0].set_title('unif nbits walk')
    for i in range(niters):
        x = unif_nbits(N=N, M=M, B=B)
        axes[0, 0].plot(x)

    axes[0, 1].set_title('gauss rand walk')
    for i in range(niters):
        x = np.cumsum(np.random.randn(N))
        axes[0, 1].plot(x)

    plt.show()


if __name__ == '__main__':
    main()
