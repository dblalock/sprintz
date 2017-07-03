#!/usr/bin/env python

import itertools
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

# from .datasets import ucr
from . import datasets as ds
from .utils import files
from .utils import sliding_window as window


SAVE_DIR = 'figs/ucr/'
# SAVE_DIR += 'derivs'


def color_for_label(lbl):
    COLORS = ['b','g','r','c','m','y','k']  # noqa
    idx = lbl - 1        # class labels usually start at 1 (though sometimes 0)
    if not (0 <= idx < len(COLORS)):
        return 'k'
    return COLORS[int(idx)]


def name_from_dir(datasetDir):
    return os.path.basename(datasetDir)


def save_current_plot(datasetName, suffix='', subdir=''):
    # datasetName = name_from_dir(datasetDir)
    saveDir = os.path.join(os.path.expanduser(SAVE_DIR), subdir)
    files.ensure_dir_exists(saveDir)
    fileName = os.path.join(saveDir, datasetName) + suffix
    plt.savefig(fileName)
    plt.close()


def plot_examples(X, transforms=None, Y=None, maxPerClass=None, ax=None):

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 8))

    plot_classes = Y is not None and maxPerClass is not None
    if plot_classes:
        numPlottedForClasses = np.zeros(max(Y) + 1)

    for row in range(X.shape[0]):
        if plot_classes:
            # only plot a fixed number of examples of each
            # class so that the plots come out legible
            lbl = Y[row]
            if (numPlottedForClasses[lbl] > maxPerClass):
                continue
            numPlottedForClasses[lbl] += 1

        data = X[row, :]
        if transforms:
            for transform in transforms:
                data = transform(data)

        if plot_classes:
            ax.plot(data, color=color_for_label(lbl))
        else:
            ax.plot(data)

    # suffix = ""
    # subdir = ""

    # if maxPerClass < np.inf:
    #     suffix += "_" + str(maxPerClass)
    #     subdir += "_" + str(maxPerClass)
    # if transforms:
    #     for transform in transforms:
    #         name = '_' + str(transform.__name__)
    #         suffix += name
    #         subdir += name
    # save_current_plot(datasetDir, suffix=suffix, subdir=subdir)


def learn_dict_seq(tsList, max_len=2048):
    # seq = np.zeros(max_len)

    # TODO self pick up here
    #   -add a subseq to seq whenever it wouldn't be compressed by seq so far
    #   -update each subseq in seq whenever it's the best match
    #   -optional refinement using SAX to find approximate best matches

    # use window.<whatever> for subseqs, probably

    # generate gaussian, then laplacian random walk, std of 16 prolly
    # whenever it would go outside [0, 255], flip the sign when accumulating
    #
    subseq_len = 8

    seq = np.zeros(max_len, dtype=np.float32)
    sums = np.zeros(max_len, dtype=np.float32)
    counts = np.ones(max_len, dtype=np.float32)
    # scale_factors = np.ones(max_len, dtype=np.float32)  # for scaling dists

    # random walk, but reverse direction when it would exceed the bounds
    deltas = np.random.randn(max_len) * 8
    sign = 1
    for i in range(8, max_len):
        new_val = seq[i-1] + sign * deltas[i]
        if -128 <= new_val <= 127:
            seq[i] = new_val
        else:
            sign = -sign
            seq[i] = seq[i-1] + sign * deltas[i]

    seq_orig = np.copy(seq)
    # sums = np.copy(seq)

    # mean normalized
    dict_subseqs = window.sliding_window_1D(seq, subseq_len)
    dict_subseqs -= np.mean(dict_subseqs, axis=1, keepdims=True)
    dict_norms = np.sum(dict_subseqs*dict_subseqs, axis=1, keepdims=True)
    dict_subseqs /= np.sqrt(dict_norms)

    _, axes = plt.subplots(2, figsize=(8, 6))
    ax = axes[1]

    # for ts in tsList[:5]:
    # for i, ts in enumerate(tsList[:50]):
    for i, ts in enumerate(tsList):
        # znormalize the windows of the time series
        data_subseqs = window.sliding_window_1D(ts, subseq_len)
        data_subseqs -= np.mean(data_subseqs, axis=1, keepdims=True)
        # data_subseqs /= np.var(data_subseqs, axis=1, keepdims=True)

        # TODO self log nn MSE here and verify that it tends to go down

        for subseq in data_subseqs:
            prods = np.dot(dict_subseqs, subseq)
            # dists = -2*prods + dict_norms
            # dists *= scale_factors
            # idx = np.argmin(dists)
            idx = np.argmax(prods)
            sums[idx:idx+subseq_len] += (subseq - dict_subseqs[idx])
            counts[idx:idx+subseq_len] += 1

        diffs = sums / counts
        sums[:] = 0
        counts[:] = 1

        # seq = seq_orig + diffs
        seq += diffs

        dict_subseqs = window.sliding_window_1D(seq, subseq_len)
        dict_subseqs -= np.mean(dict_subseqs, axis=1, keepdims=True)
        dict_norms = np.sum(dict_subseqs*dict_subseqs, axis=1, keepdims=True)
        dict_subseqs /= np.sqrt(dict_norms)

        if i % 10 == 0:
            # plt.plot(seq)
            ax.plot(diffs)

    # axes[0].plot(seq_orig, 'b')
    axes[0].plot(seq, 'g')

    # convert signed to unsigned
    # seq += 128

    # # seq = np.random.laplace(scale=16., size=max_len)
    # seq = np.random.laplace(size=max_len)
    # seq = np.cumsum(seq)

    # plt.plot(seq)
    # plt.plot(np.cumsum(np.random.randn(max_len)))
    # seq_smooth = np.cumsum(np.random.laplace(size=(max_len * 2)))
    # seq_smooth = np.mean(seq_smooth.reshape((-1, 2)), axis=1)
    # plt.plot(seq_smooth)
    plt.show()


DEFAULT_MAX_NN_IDX = (1 << 11) - 1


def nn_encode(diffs, num_neighbors, nn_step=4, hash_algo=None):

    if hash_algo is None or hash_algo == 'none':
        assert nn_step == 4  # we only handle 4 for now
        diff_windows = window.sliding_window_1D(diffs.ravel(), 8, step=8)
        offsetBlocks = np.zeros_like(diff_windows)

        zero_samples = np.zeros(8, dtype=np.int32)
        N = len(diff_windows)

        for n in range(1, N):
            start_idx = int(max(0, n - (nn_step / 8.) * num_neighbors))
            end_idx = n
            history = diff_windows[start_idx:end_idx].ravel()
            nn_windows = window.sliding_window_1D(history, 8, step=4)
            neighbors = np.vstack((nn_windows, zero_samples))

            costs = np.abs(neighbors - diff_windows[n])
            costs = np.max(costs, axis=1)
            nn_idx = np.argmin(costs)
            offsetBlocks[n] = diff_windows[n] - neighbors[nn_idx]

        return offsetBlocks

    if hash_algo == 'perm':
        assert nn_step == 4  # we only handle 4 for now
        diff_windows = window.sliding_window_1D(diffs.ravel(), 8, step=8)
        # nn_windows = window.sliding_window_1D(diffs.ravel(), 8, step=4)
        offsetBlocks = np.zeros_like(diff_windows)

        zero_samples = np.zeros(8, dtype=np.int32)
        N = len(diff_windows)


        # SELF: pick up here


    raise ValueError("Unrecognized hash_algo: '{}'".format(hash_algo))


def quantize_diff_block(X, numbits, keep_nrows=100):
    maxval = (1 << numbits) - 1
    X = X[:100]
    X = X - np.min(X)
    X = (maxval / float(np.max(X)) * X).astype(np.int32)
    diffs = X[:, 1:] - X[:, :-1]

    tail = diffs.size % 8
    blocks = diffs.ravel()[:-tail] if tail > 0 else diffs.ravel()
    blocks = blocks.reshape((-1, 8))

    return X, diffs, blocks


def plot_dset(d, numbits=8, offset_algo='minimax', nn_step=4,
              num_neighbors=DEFAULT_MAX_NN_IDX):

    maxval = (1 << numbits) - 1
    _, axes = plt.subplots(4, 2, figsize=(10, 8))
    # axes = np.concat()
    axes = np.concatenate([axes[:, 0], axes[:, 1]])
    # axes = zip(axes)[0] + zip(axes)[1]  # make 1d, first col then second col
    # print axes
    # return

    # ------------------------ data munging

    # X = d.X[:100]
    # X = X - np.min(X)
    # X = (maxval / float(np.max(X)) * X).astype(np.int32)
    # # axes[1].set_ylim((0, 255))

    # diffs = X[:, 1:] - X[:, :-1]
    # # absDiffs = np.abs(diffs)
    # tail = diffs.size % 8

    # blocks = diffs.ravel()[:-tail] if tail > 0 else diffs.ravel()
    # blocks = blocks.reshape((-1, 8))

    X, diffs, blocks = quantize_diff_block(d.X, numbits, )
    absBlocks = np.abs(blocks)

    if offset_algo == 'minimax':
        mins = np.min(blocks, axis=1)
        ranges = np.max(blocks, axis=1) - mins
        sub_vals = (mins + ranges / 2).astype(np.int32)
        offsetBlocks = blocks - sub_vals.reshape((-1, 1))
    elif offset_algo == 'nn7':
        diff_windows = window.sliding_window_1D(diffs.ravel(), 8, step=8)
        sub_windows = np.zeros_like(diff_windows)
        N = len(diff_windows)

        zero_samples = np.zeros(8, dtype=np.int32)
        for n in range(6 * nn_step + 8, N):
            assert nn_step == 4  # below only handles this value
            neighbors_slice = diff_windows[(n-4):n].ravel()
            neighbors = window.sliding_window_1D(neighbors_slice, 8, step=4)
            neighbors = np.vstack((neighbors, zero_samples))
            assert len(neighbors) == 8

            costs = np.abs(neighbors - diff_windows[n])
            costs = np.max(costs, axis=1)
            nn_idx = np.argmin(costs)
            sub_windows[n] = neighbors[nn_idx]

        offsetBlocks = blocks - sub_windows

    elif offset_algo == 'nn':
        offsetBlocks = nn_encode(diffs, num_neighbors)

        # assert nn_step == 4  # we only handle 4 for now
        # diff_windows = window.sliding_window_1D(diffs.ravel(), 8, step=8)
        # # nn_windows = window.sliding_window_1D(diffs.ravel(), 8, step=4)
        # offsetBlocks = np.zeros_like(diff_windows)

        # zero_samples = np.zeros(8, dtype=np.int32)
        # N = len(diff_windows)

        # for n in range(1, N):
        #     start_idx = int(max(0, n - (nn_step / 8.) * num_neighbors))
        #     end_idx = n
        #     history = diff_windows[start_idx:end_idx].ravel()
        #     nn_windows = window.sliding_window_1D(history, 8, step=4)
        #     neighbors = np.vstack((nn_windows, zero_samples))

        #     costs = np.abs(neighbors - diff_windows[n])
        #     costs = np.max(costs, axis=1)
        #     nn_idx = np.argmin(costs)
        #     offsetBlocks[n] = diff_windows[n] - neighbors[nn_idx]

    else:
        raise ValueError("Unrecognized offset algorithm: {}".format(
            offset_algo))

    absOffsetBlocks = np.abs(offsetBlocks)
    absDiffs = np.abs(absBlocks).ravel()
    absOffsetDiffs = absOffsetBlocks.ravel()

    # "examples" stitched together from blocks with vals subbed off
    use_shape = diffs.shape[0] - 1, diffs.shape[1]
    use_size = use_shape[0] * use_shape[1]
    diffs_offset = offsetBlocks.ravel()[:use_size].reshape(use_shape)

    def compute_signed_overflow(resids, nbits):
        cutoff = int(2**(nbits - 1)) - 1
        x_bad = np.where(np.abs(resids).ravel() > cutoff)[0]
        y_bad = resids.ravel()[x_bad]
        x_bad = np.mod(x_bad, resids.shape[1])
        overflow_frac = len(x_bad) / float(resids.size)
        return x_bad, y_bad, cutoff, overflow_frac

    x_bad, y_bad, _, overflow_frac = \
        compute_signed_overflow(diffs, numbits)
    x_bad4, y_bad4, cutoff, overflow_frac4 = \
        compute_signed_overflow(diffs, numbits - 4)

    x_bad_offset, y_bad_offset, _, overflow_frac_offset = \
        compute_signed_overflow(diffs_offset, numbits)
    x_bad4_offset, y_bad4_offset, _, overflow_frac4_offset = \
        compute_signed_overflow(diffs_offset, numbits - 4)

    # ------------------------ actual plotting

    axes[0].set_title("{} ({} bits)".format(d.name, numbits))

    axes[1].set_title("{} deltas ({:.2f}% overflow)".format(
        d.name, 100 * overflow_frac))
    axes[-3].set_title("{} offset deltas ({:.2f}% overflow)".format(
        d.name, 100 * overflow_frac_offset))

    axes[2].set_title("Deltas ({:.2f}% {}bit overflow)".format(
        100 * overflow_frac4, int(np.log2(cutoff + 1) + 1)))

    axes[-2].set_title("Deltas after offseting ({:.2f}% {}bit overflow)".format(
        100 * overflow_frac4_offset, int(np.log2(cutoff + 1) + 1)))

    axes[3].set_title("PDF/CDF of |max block delta| (with nbits lines)")
    axes[-1].set_title("PDF/CDF of |max block delta| after offseting")

    axes[0].set_ylim((0, maxval))
    axes[1].set_ylim((-maxval/2, maxval/2))
    axes[2].set_ylim((-cutoff - 1, cutoff))
    axes[-2].set_ylim((-cutoff - 1, cutoff))

    # plot raw data
    plot_examples(X, ax=axes[0])

    # plot diffs
    plot_examples(diffs, ax=axes[1])
    axes[1].scatter(x_bad, y_bad, s=np.pi * 10*10,
                    facecolors='none', edgecolors='r')
    # plot offset diffs
    plot_examples(diffs_offset, ax=axes[-3])
    axes[-3].scatter(x_bad_offset, y_bad_offset, s=np.pi * 10*10,
                     facecolors='none', edgecolors='r')
    axes[-3].set_ylim(axes[1].get_ylim())

    # plot how well data fits in a smaller window
    plot_examples(diffs, ax=axes[2])
    axes[2].scatter(x_bad4, y_bad4, s=np.pi * 10*10,
                    facecolors='none', edgecolors='r')

    # plot blocks after subtracting values at end of previous blocks
    plot_examples(diffs_offset, ax=axes[-2])
    axes[-2].scatter(x_bad4_offset, y_bad4_offset, s=np.pi * 10*10,
                     facecolors='none', edgecolors='r')

    def plot_uresiduals_bits(resids, ax):

        sb.distplot(resids, ax=ax, kde=False, hist_kws={'normed': True})

        ax.set_xlim((0, min(np.max(resids), 256)))
        # ax = ax.twinx()
        # ax.set_xlim(axes[3].get_xlim())
        # ax.set_ylim((0, 1))
        # sb.distplot(np.max(absBlocks, axis=1), ax=ax, hist=False, color='brown',
        # sb.distplot(np.max(resids, axis=1), ax=ax, hist=False, color='brown',
        #             kde_kws={'cumulative': True})

        _, ymax = ax.get_ylim()
        ax.set_ylim([0, ymax])
        cutoffs = [4, 8, 16, 32, 64]
        max_nbits_left = ax.get_xlim()[1]
        if max_nbits_left > 128:
            cutoffs.append(128)
        for i, cutoff in enumerate(cutoffs):
            # ax.plot([cutoff, cutoff], [0, ymax], 'k--', lw=1)
            ax.plot([cutoff, cutoff], [0, 1], 'k--', lw=1)
            x = cutoff / float(max_nbits_left)
            y = .05 + (.8 * (i % 2)) + (.025 * (i % 4))  # stagger heights
            nbits = str(int(np.log2(cutoff) + 1))
            ax.annotate(nbits, xy=(x, y), xycoords='axes fraction')

    plot_uresiduals_bits(absDiffs, ax=axes[3])
    plot_uresiduals_bits(absOffsetDiffs, ax=axes[-1])

    _, ymax_left = axes[3].get_ylim()
    _, ymax_right = axes[-1].get_ylim()
    ymax = max(ymax_left, ymax_right)
    [ax.set_ylim([0, ymax]) for ax in (axes[3], axes[-1])]

    plt.tight_layout()


def create_perm_lut():
    lut = np.zeros(64, dtype=np.uint8) + 255

    def perm_number(perm):
        # comparisons order:
        #   (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        #
        # I'm using > instead of >= because I'm more certain that there are
        # only 24 combos of these, since strict inequality is never symmetric
        ret = 0
        if perm[0] > perm[1]:
            ret += 32
        if perm[0] > perm[2]:
            ret += 16
        if perm[0] > perm[3]:
            ret += 8
        if perm[1] > perm[2]:
            ret += 4
        if perm[1] > perm[3]:
            ret += 2
        if perm[2] > perm[3]:
            ret += 1

        return ret

    perms = itertools.permutations(range(4))
    perms = np.array(list(perms))  # since we'll return it
    for i, perm in enumerate(perms):
        lut[perm_number(perm)] = i

    # this returns the lut:
    # [  0   1 255   3   2 255   4   5 255 255 255   9 255 255 255  11 255 255
    # 255 255   8 255  10 255 255 255 255 255 255 255  16  17   6   7 255 255
    # 255 255 255 255 255  13 255  15 255 255 255 255  12 255 255 255  14 255
    # 255 255  18  19 255  21  20 255  22  23]
    #
    # Note that, to be super safe, should use 0 instead of 255 as placeholder
    return lut, perms


def hash_block(samples, lut, perms):
    """maps 8 samples to an index in {0,1,...,31}

    Assumes latter params are from create_perm_lut()"""

    sums = samples.reshape((4, 2)).sum(axis=1)
    idx = int(sums[0] > sums[1]) << 5
    idx += (sums[0] > sums[2]) << 4
    idx += (sums[0] > sums[3]) << 3
    idx += (sums[1] > sums[2]) << 2
    idx += (sums[1] > sums[3]) << 1
    idx += (sums[2] > sums[3])

    perm_idx = lut[idx]

    # if largest/smallest sum much bigger than others, use that
    perm = perms[perm_idx]
    if sums[perm[0]] > (sums[perm[1]] << 1):
        return 24 + perm[0]
    if sums[perm[3]] < (sums[perm[2]] << 1):
        return 28 + perm[3]

    # otherwise, return bucket based on overall shape
    return perm_idx


def plot_hash_func(d, numbits, num_neighbors=256, nn_step=4):
    pass


def main():
    # # # uncomment this to get the LUT for mapping comparison bytes to indices
    # lut, perms = create_perm_lut()
    # print "perm lut, perms: ", lut, "\n", perms
    # assert len(np.unique(lut)) == 25
    # import sys
    # sys.exit()

    # dsets = ds.smallUCRDatasets()

    # dset = list(dsets)[2]
    # X = dset.X

    # print "learning dict seq for dataset: ", dset.name
    # learn_dict_seq(X, max_len=1024)

    # import sys
    # sys.exit()

    # mpl.rcParams.update({'figure.autolayout': True})  # make twinx() not suck

    # dsets = ds.allUCRDatasets()
    dsets = ds.smallUCRDatasets()

    # numbits = 16
    numbits = 12
    # numbits = 8

    # print [d.name for d in dsets]
    # for d in list(dsets)[2:3]:
    # for d in list(dsets)[9:10]:
    for d in dsets:
        # plot_dset(d, numbits=numbits, offset_algo='nn7')
        plot_dset(d, numbits=numbits, offset_algo='nn')
        save_current_plot(d.name, subdir='small/deltas/{}b_nn'.format(numbits))
        # save_current_plot(d.name, subdir='small/deltas/{}b_nn7'.format(numbits))
        # save_current_plot(d.name, subdir='small/deltas/{}b'.format(numbits))
        # save_current_plot(d.name, subdir='deltas/{}b_nn'.format(numbits))
        # save_current_plot(d.name, subdir='deltas/{}b_nn7'.format(numbits))
        # save_current_plot(d.name, subdir='deltas/{}b'.format(numbits))

    # plt.show()


if __name__ == '__main__':
    main()
