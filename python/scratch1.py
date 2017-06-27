#!/usr/bin/env python

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


def plot_dset(d, numbits=8):

    maxval = (1 << numbits) - 1
    _, axes = plt.subplots(4, 2, figsize=(10, 8))
    # axes = np.concat()
    axes = np.concatenate([axes[:, 0], axes[:, 1]])
    # axes = zip(axes)[0] + zip(axes)[1]  # make 1d, first col then second col
    # print axes
    # return

    # ------------------------ data munging

    X = d.X[:100]
    X = X - np.min(X)
    X = (maxval / float(np.max(X)) * X).astype(np.int32)
    # axes[1].set_ylim((0, 255))

    diffs = X[:, 1:] - X[:, :-1]
    # absDiffs = np.abs(diffs)
    tail = diffs.size % 8

    blocks = diffs.ravel()[:-tail] if tail > 0 else diffs.ravel()
    blocks = blocks.reshape((-1, 8))
    absBlocks = np.abs(blocks)

    prev_vals = np.insert(blocks[:-1, -1], 0, 0).reshape((-1, 1))
    offsetBlocks = blocks - prev_vals
    absOffsetBlocks = np.abs(offsetBlocks)

    absDiffs = np.abs(absBlocks).ravel()
    absOffsetDiffs = absOffsetBlocks.ravel()

    # "examples" stitched together from blocks with prev vals subbed off
    use_shape = diffs.shape[0] - 1, diffs.shape[1]
    use_size = use_shape[0] * use_shape[1]
    diffs_offset = offsetBlocks.ravel()[:use_size].reshape(use_shape)

    # print "diffs.shape", absBlocks.shape
    # print "absDiffs.shape", absBlocks.shape
    # print "tail len", tail
    # print "absBlocks.shape", absBlocks.shape
    # return

    # 8bit overflow
    # x_bad = np.where(absDiffs > int(maxval / 2))[0]
    # y_bad = diffs.ravel()[x_bad]
    # x_bad = np.mod(x_bad, diffs.shape[1])
    # overflow_frac = len(x_bad) / float(X.size)

    # 4bit overflow (or 8bit for 16bit values)
    # cutoff = int(2**(numbits - 5)) - 1
    # x_bad4 = np.where(absDiffs > cutoff)[0]
    # y_bad4 = diffs.ravel()[x_bad4]
    # x_bad4 = np.mod(x_bad4, diffs.shape[1])
    # overflow_frac4 = len(x_bad4) / float(X.size)

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

    axes[-2].set_title("Deltas after subbing prev delta ({:.2f}% {}bit overflow)".format(
        100 * overflow_frac4_offset, int(np.log2(cutoff + 1) + 1)))

    axes[3].set_title("PDF/CDF of |max block delta| (with nbits lines)")
    axes[-1].set_title("PDF/CDF of |max block delta| after subbing prev delta")

    axes[0].set_ylim((0, maxval))
    axes[1].set_ylim((-maxval/2, maxval/2))
    axes[2].set_ylim((-cutoff - 1, cutoff))
    axes[-2].set_ylim((-cutoff - 1, cutoff))
    # axes[3].set_ylim((0, .5))

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

        sb.distplot(resids, ax=ax)

        ax.set_xlim((0, min(np.max(resids), 256)))
        ax = ax.twinx()
        ax.set_xlim(axes[3].get_xlim())
        ax.set_ylim((0, 1))
        sb.distplot(np.max(absBlocks, axis=1), ax=ax, hist=False, color='brown',
                    kde_kws={'cumulative': True})

        # cutoffs = [2, 4, 8, 16, 32, 64]
        cutoffs = [4, 8, 16, 32, 64]
        max_nbits_left = ax.get_xlim()[1]
        if max_nbits_left > 128:
            cutoffs.append(128)
        for i, cutoff in enumerate(cutoffs):
            ax.plot([cutoff, cutoff], [0, 1], 'k--', lw=1)
            x = cutoff / float(max_nbits_left)
            y = .05 + (.8 * (i % 2)) + (.025 * (i % 4))  # stagger heights
            nbits = str(int(np.log2(cutoff) + 1))
            ax.annotate(nbits, xy=(x, y), xycoords='axes fraction')

    plot_uresiduals_bits(absDiffs, ax=axes[3])
    plot_uresiduals_bits(absOffsetDiffs, ax=axes[-1])

    plt.tight_layout()


def main():
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
    # numbits = 12
    numbits = 8

    # print [d.name for d in dsets]
    # for d in list(dsets)[2:3]:
    # for d in list(dsets)[9:10]:
    for d in dsets:
        plot_dset(d, numbits=numbits)
        save_current_plot(d.name, subdir='small/deltas/{}b'.format(numbits))
        # save_current_plot(d.name, subdir='deltas/{}b'.format(numbits))

        # # if d.name not in ('Two_Patterns', 'synthetic_control'):  # TODO rm
        # #     continue

        # # _, axes = plt.subplots(3, figsize=(6, 8))
        # _, axes = plt.subplots(4, figsize=(6, 8))

        # # ------------------------ data munging

        # X = d.X[:100]
        # X = X - np.min(X)
        # X = (255. / np.max(X) * X).astype(np.int32)
        # # axes[1].set_ylim((0, 255))

        # diffs = X[:, 1:] - X[:, :-1]
        # absDiffs = np.abs(diffs)

        # # 8bit overflow
        # x_bad = np.where(absDiffs.ravel() > 127)[0]
        # y_bad = diffs.ravel()[x_bad]
        # x_bad = np.mod(x_bad, diffs.shape[1])
        # overflow_frac = len(x_bad) / float(X.size)

        # # 4bit overflow
        # x_bad4 = np.where(absDiffs.ravel() > 7)[0]
        # y_bad4 = diffs.ravel()[x_bad4]
        # x_bad4 = np.mod(x_bad4, diffs.shape[1])
        # overflow_frac4 = len(x_bad4) / float(X.size)

        # # ------------------------ actual plotting

        # axes[0].set_title(d.name)
        # axes[1].set_title("{} deltas ({:.2f}% overflow)".format(
        #     d.name, 100 * overflow_frac))
        # axes[2].set_title("{} deltas ({:.2f}% 4bit overflow)".format(
        #     d.name, 100 * overflow_frac4))
        # axes[3].set_title("PDF/CDF of {} |deltas| (with nbits lines)".format(d.name))

        # axes[0].set_ylim((0, 255))
        # axes[1].set_ylim((-127, 127))
        # axes[2].set_ylim((-8, 7))
        # axes[3].set_xlim((0, np.max(absDiffs)))
        # # axes[3].set_ylim((0, .5))

        # plot_examples(X, ax=axes[0])

        # plot_examples(diffs, ax=axes[1])
        # axes[1].scatter(x_bad, y_bad, s=np.pi * 10*10,
        #                 facecolors='none', edgecolors='r')

        # plot_examples(diffs, ax=axes[2])
        # axes[2].scatter(x_bad4, y_bad4, s=np.pi * 10*10,
        #                 facecolors='none', edgecolors='r')

        # sb.distplot(absDiffs.ravel(), ax=axes[3])
        # # axes[3].hist(absDiffs.ravel())
        # # sb.distplot(absDiffs.ravel(), kde=False, ax=axes[3])
        # # sb.distplot(absDiffs.ravel(), ax=axes[3],
        #             # hist_kws=dict(cumulative=True),
        #             # kde_kws=dict(cumulative=True))

        # ax = axes[3].twinx()
        # ax.set_xlim(axes[3].get_xlim())
        # ax.set_ylim((0, 1))
        # sb.distplot(absDiffs.ravel(), ax=ax, hist=False, color='brown',
        #             kde_kws={'cumulative': True})

        # # cutoffs = [2, 4, 8, 16, 32, 64]
        # cutoffs = [4, 8, 16, 32, 64]
        # max_nbits_left = ax.get_xlim()[1]
        # if max_nbits_left > 128:
        #     cutoffs.append(128)
        # for i, cutoff in enumerate(cutoffs):
        #     ax.plot([cutoff, cutoff], [0, 1], 'k--', lw=1)
        #     x = cutoff / float(max_nbits_left)
        #     y = .05 + (.8 * (i % 2)) + (.025 * (i % 4))  # stagger heights
        #     nbits = str(int(np.log2(cutoff) + 1))
        #     ax.annotate(nbits, xy=(x, y), xycoords='axes fraction')

        # plt.tight_layout()
        # save_current_plot(d.name, subdir='small/deltas')
        # save_current_plot(d.name, subdir='deltas')
    # plt.show()


if __name__ == '__main__':
    main()
