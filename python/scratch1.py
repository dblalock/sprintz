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


def imshow_better(X, ax=None, cmap=None):
    if not ax:
        ax = plt
    return ax.imshow(X, interpolation='nearest', aspect='auto', cmap=cmap)


def color_for_label(lbl):
    COLORS = ['b','g','r','c','m','y','k']  # noqa
    idx = lbl - 1        # class labels usually start at 1 (though sometimes 0)
    if not (0 <= idx < len(COLORS)):
        return 'k'
    return COLORS[int(idx)]


def name_from_dir(datasetDir):
    return os.path.basename(datasetDir)


# def save_current_plot(datasetName, suffix='.pdf', subdir=''):
def save_current_plot(datasetName, suffix='.png', subdir=''):
    # datasetName = name_from_dir(datasetDir)
    saveDir = os.path.join(os.path.expanduser(SAVE_DIR), subdir)
    files.ensure_dir_exists(saveDir)
    fileName = os.path.join(saveDir, datasetName) + suffix
    plt.savefig(fileName)
    plt.close()


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


def hash_block_perm(samples, lut, perms):
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

        lw = 1
        if plot_classes:
            ax.plot(data, color=color_for_label(lbl), lw=lw)
        else:
            ax.plot(data, lw=lw)

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


def nbits_cost(diffs):
    """
    >>> [nbits_cost(i) for i in [0, 1, 2, 3, 4, 5, 7, 8, 9]]
    [0, 2, 3, 3, 4, 4, 4, 5, 5]
    >>> [nbits_cost(i) for i in [-1, -2, -3, -4, -5, -7, -8, -9]]
    [1, 2, 3, 3, 4, 4, 4, 5]
    >>> nbits_cost([])
    array([], dtype=int32)
    >>> nbits_cost([0, 2, 1, 0])
    array([0, 3, 2, 0], dtype=int32)
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
    return nbits.reshape(shape) if nbits.size > 1 else nbits[0]  # unpack if scalar


def block_nbits_costs(blocks):
    """max nbits cost for each row"""
    N, _ = blocks.shape
    blocks = np.abs(blocks) + (blocks >= 0).astype(np.int32)
    maxes = np.max(blocks, axis=1)
    return nbits_cost(maxes).reshape((N, 1))


# def nn_encode(diffs, num_neighbors, nn_step=4, hash_algo=None):
def nn_encode(blocks, num_neighbors=256, nn_step=4, hash_algo=None,
              scale_factors=None, try_invert=True):

    if scale_factors is not None and 1 in scale_factors:
        scale_factors = scale_factors[:]
        scale_factors.remove(1)

    if hash_algo is None or hash_algo == 'none':
        # assert nn_step == 4  # we only handle 4 for now
        # diff_windows = window.sliding_window_1D(diffs.ravel(), 8, step=8)
        # offsetBlocks = np.zeros_like(diff_windows)
        offsetBlocks = np.copy(blocks)

        zero_samples = np.zeros(8, dtype=np.int32)
        # N = len(diff_windows)
        N = len(blocks)

        times_nonzero = 0
        saved_bits = 0
        saved_bits_sq = 0
        for n in range(1, N):
            # start_idx = int(max(0, n - (nn_step / 8.) * num_neighbors))
            # end_idx = n
            # history = diff_windows[start_idx:end_idx].ravel()
            # nn_windows = window.sliding_window_1D(history, 8, step=4)
            # neighbors = np.vstack((nn_windows, zero_samples))

            # this isn't exactly where we'd start in the flattened array,
            # but whatever
            start_idx = n - 1 - ((num_neighbors - 1) * nn_step / 8)
            start_idx = max(0, start_idx)
            end_idx = n
            history = blocks[start_idx:end_idx].ravel()

            nn_windows = window.sliding_window_1D(history, 8, step=nn_step)
            all_neighbors = [nn_windows]

            if try_invert:
                all_neighbors.append(-nn_windows)

            if scale_factors is not None:
                add_neighbors = []
                for scale in scale_factors:
                    for neighbormat in all_neighbors:
                        # scaled_mat = (neighbormat << scale[0]) >> scale[1]
                        scaled_mat = (neighbormat * scale).astype(np.int32)
                        add_neighbors.append(scaled_mat)
                all_neighbors += add_neighbors

            all_neighbors = [zero_samples] + all_neighbors
            neighbors = np.vstack(all_neighbors)

            costs = np.abs(neighbors - blocks[n])
            costs = np.max(costs, axis=1)
            nn_idx = np.argmin(costs)

            nn_cost_bits = nbits_cost(costs[nn_idx])
            zeros_cost_bits = nbits_cost(costs[0])
            bitsave = zeros_cost_bits - nn_cost_bits
            saved_bits += bitsave
            saved_bits_sq += bitsave*bitsave

            offsetBlocks[n] -= neighbors[nn_idx]

            times_nonzero += int(nn_idx > 0)

        N_f = float(N)
        expected_bitsave = saved_bits / N_f
        std_bitsave = saved_bits_sq / N_f - expected_bitsave * expected_bitsave
        print "nn nonzero {}/{} ({:.1f}%) and saved {:.2f} +/-{:.2f} bits" \
            .format(times_nonzero, N, 100 * times_nonzero / N_f,
                    expected_bitsave, std_bitsave)

        return offsetBlocks

        # TODO option to use various hash funcs here

    raise ValueError("Unrecognized hash_algo: '{}'".format(hash_algo))


def convert_to_blocks(diffs):
    tail = diffs.size % 8
    blocks = diffs.ravel()[:-tail] if tail > 0 else diffs.ravel()
    return blocks.reshape((-1, 8))


def quantize_diff_block(X, numbits, keep_nrows=100):
    maxval = (1 << numbits) - 1
    X = X[:keep_nrows]
    X = X - np.min(X)
    X = (maxval / float(np.max(X)) * X).astype(np.int32)
    diffs = X[:, 1:] - X[:, :-1]

    blocks = convert_to_blocks(diffs)

    return X, diffs, blocks


# def _sign_scales(nbits_costs):
#     """
#     >>> nbits = np.array([[0, 0], [1, 0], [2]])
#     >>> _sign_scales(nbits)
#     np.array([-1, ])

#     """
#     pass


def scaled_signs_encode(blocks):
    """
    >>> blocks = np.array([[0, 0], [1, 0], [2]])

    """
    # blocks_gt0 = blocks >= 0
    signs = ((blocks >= 0) << 1) - 1  # >=0 -> +1; <0 -> -1
    nbits = block_nbits_costs(blocks)  # max nbits for each row
    nbits = np.minimum(nbits, 7)

    # nbits     meaning
    # 0         all 0s; no offset
    # 1         worst is -1; just sub the sign, but 0 needs to get 0 subbed
    # 2         [-2, 1], so just sub the sign (shift by 0)
    # 3         [-4, 3], so sub 2 * sign (shift by 1)
    scales_shifts = np.maximum(nbits - 2, 0)

    offsets = (signs << scales_shifts) * (scales_shifts > 0)
    offsets -= (signs > 0) * (nbits == 1)  # don't sub +1 when stuff in [-1, 0]
    return blocks - offsets


def my_transform_orig(blocks):
    offsetBlocks2 = np.empty(blocks.shape, dtype=np.int32)
    for i in range(1, blocks.shape[1], 2):
        offsetBlocks2[:, i] = blocks[:, i-1] >> 1 + blocks[:, i] >> 1
    offsetBlocks2[:, 0] = blocks[:, 0] - offsetBlocks2[:, 1]
    return offsetBlocks2


def my_old_transform(blocks):  # old version that loses LSBs
    offsetBlocks2 = np.empty(blocks.shape, dtype=np.int32)
    for i in range(1, blocks.shape[1], 2):
        offsetBlocks2[:, i] = blocks[:, i-1] >> 1 + blocks[:, i] >> 1 + \
            np.bitwise_and(blocks[:, i], 1)
        offsetBlocks2[:, i-1] = blocks[:, i-1] - offsetBlocks2[:, i]
    return offsetBlocks2


def my_transform(blocks):
    blocks = np.copy(blocks)
    for i in range(0, blocks.shape[1], 2):
        blocks[:, 1] -= blocks[:, 0] >> 1
        blocks[:, 0] -= blocks[:, 1] >> 1
    return blocks


def my_transform_inverse(blocks):
    blocks = np.copy(blocks)
    for i in range(0, blocks.shape[1], 2):
        blocks[:, 0] += blocks[:, 1] >> 1
        blocks[:, 1] += blocks[:, 0] >> 1
    return blocks


def inflection_encode(blocks):
    shape = blocks.shape
    blocks = np.copy(blocks.ravel())
    for i in range(1, len(blocks)):
        pass

def plot_dset(d, numbits=8, offset_algo='center', n=100, **nn_kwargs):

    if offset_algo is None or isinstance(offset_algo, str):
        offset_algo = (offset_algo,)  # it's a collection

    _, axes = plt.subplots(4, 2, figsize=(10, 8))
    axes = np.concatenate([axes[:, 0], axes[:, 1]])

    # ------------------------ data munging

    X, diffs, blocks = quantize_diff_block(d.X, numbits, keep_nrows=n)

    # first round of transforms; defines offsetBlocks var
    if 'double_delta' in offset_algo:  # delta encode deltas
        double_diffs = np.copy(diffs)
        double_diffs[:, 1:] = diffs[:, 1:] - diffs[:, :-1]
        offsetBlocks = convert_to_blocks(double_diffs)

    elif '1.5_delta' in offset_algo:  # sub only half of previous delta
        semi_diffs = np.copy(diffs)
        semi_diffs[:, 1:] = diffs[:, 1:] - (diffs[:, :-1] >> 1)
        offsetBlocks = convert_to_blocks(semi_diffs)

    elif 'dyn_delta' in offset_algo:  # pick delta or delta-delta
        double_diffs = np.copy(diffs)
        double_diffs[:, 1:] = diffs[:, 1:] - diffs[:, :-1]
        blocks_dbl = convert_to_blocks(double_diffs)

        costs_single = np.max(np.abs(blocks), axis=1)
        costs_dbl = np.max(np.abs(blocks_dbl), axis=1)
        better_idxs = costs_dbl < costs_single

        offsetBlocks = np.copy(blocks)
        offsetBlocks[better_idxs] = blocks_dbl[better_idxs]
    else:
        offsetBlocks = np.copy(blocks)

    if 'center' in offset_algo:
        mins = np.min(offsetBlocks, axis=1)
        ranges = np.max(offsetBlocks, axis=1) - mins
        sub_vals = (mins + ranges / 2).astype(np.int32)
        offsetBlocks -= sub_vals.reshape((-1, 1))

    if 'scaled_signs' in offset_algo:
        offsetBlocks = scaled_signs_encode(offsetBlocks)

    if 'nn' in offset_algo:
        offsetBlocks = nn_encode(offsetBlocks, **nn_kwargs)

    if 'avg' in offset_algo:
        offsetBlocks2 = np.empty(offsetBlocks.shape, dtype=np.int32)
        for i in range(1, 8):
            offsetBlocks2[:, i] = offsetBlocks[:, i-1] >> 1 + offsetBlocks[:, i] >> 1
        offsetBlocks2[:, 0] = offsetBlocks[:, 0] - offsetBlocks2[:, 1]
        offsetBlocks = np.copy(offsetBlocks2)

    offset_algo_copy = [el for el in offset_algo]
    while 'mine' in offset_algo_copy:
        offsetBlocks = my_transform(offsetBlocks)
        offset_algo_copy.remove('mine')
        if 'mine' in offset_algo_copy:
            perm = [1, 2, 5, 6, 7, 4, 3, 0]
            offsetBlocks = offsetBlocks[:, perm]

    # "examples" stitched together from blocks with vals subbed off
    # if diffs_offset is None:
    use_shape = diffs.shape[0] - 1, diffs.shape[1]
    use_size = use_shape[0] * use_shape[1]
    diffs_offset = offsetBlocks.ravel()[:use_size].reshape(use_shape)

    # nbits_blocks = np.max(nbits_cost(offsetBlocks), axis=1)
    # TODO

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

    axes[-2].set_title("Deltas after offsetting ({:.2f}% {}bit overflow)".format(
        100 * overflow_frac4_offset, int(np.log2(cutoff + 1) + 1)))

    axes[3].set_title("PDF of delta bits, max block bits")
    axes[-1].set_title("PDF of deltas, max block delta after offseting")

    axes[-4].set_title("Number of bits required for each sample in each ts")

    maxval = (1 << numbits) - 1
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

    # plot numbers of bits taken by each resid
    resids_nbits = nbits_cost(diffs_offset)
    # imshow_better(resids_nbits, ax=axes[-4], cmap='Blues')
    img = imshow_better(resids_nbits, ax=axes[-4], cmap='gnuplot2')
    plt.colorbar(img, ax=axes[-4])

    def plot_uresiduals_bits(resids, ax):
        # have final bar in the histogram contain everything that would
        # overflow 8b
        # max_resid = 128
        # resids = np.minimum(max_resid, resids)
        # bins = [1, 2, 4, 8, 16, 32, 127, max_resid + 1]

        max_nbits = 16
        resids = nbits_cost(resids).ravel()
        resids = np.minimum(resids, max_nbits)
        bins = np.arange(max_nbits + 1)

        sb.distplot(resids, ax=ax, kde=False, bins=bins, hist_kws={
            # 'normed': True, 'range': [0, max_resid]})
            # 'normed': True})
            'normed': True, 'range': [0, max_nbits]})
        resid_blocks = convert_to_blocks(resids)
        block_maxes = np.max(resid_blocks, axis=1)
        # sb.distplot(block_maxes, ax=ax, kde=False, bins=nbins, hist_kws={
        sb.distplot(block_maxes, ax=ax, kde=False, bins=bins, hist_kws={
            # 'normed': True, 'range': [0, max_resid], 'color': (.8, 0, 0, .05)})
            # 'normed': True, 'color': (.8, 0, 0, .05)})
            'normed': True, 'range': [0, max_nbits], 'color': (.8, 0, 0, .05)})

        # ax.set_xlim((0, min(np.max(resids), 256)))
        xlim = (0, max_nbits + 1)
        ax.set_xlim(xlim)
        ax.set_ylim((0, 1))

        # ax = ax.twinx()
        # ax.set_xlim(axes[3].get_xlim())
        # ax.set_ylim((0, 1))
        # sb.distplot(np.max(absBlocks, axis=1), ax=ax, hist=False, color='brown',
        # sb.distplot(np.max(resids, axis=1), ax=ax, hist=False, color='brown',
        #             kde_kws={'cumulative': True})

        # plot + annotate avg number of bits needed
        y = 8.95
        costs = (np.mean(resids), np.mean(block_maxes))
        styles = ('b--', 'r--')
        for i in range(2):
            cost = costs[i]
            style = styles[i]
            ax.plot([cost, cost], [0, 1], style, lw=1)
            x = cost / float(xlim[1])
            # y = .05 + (.8 * (i % 2)) + (.025 * (i % 4))  # stagger heights
            y = .78 + (.12 * (i % 2))  # stagger heights
            lbl = "{:.1f}".format(cost)
            ax.annotate(lbl, xy=(x, y), xycoords='axes fraction')

        #
        # plot cutoff for numbers of bits
        #
        # _, ymax = ax.get_ylim()
        # ax.set_ylim([0, ymax])
        # # cutoffs = [4, 8, 16, 32, 64]
        # cutoffs = [4, 8, 16, 32, 64, 126.5]  # not 128 for legibility
        # cutoffs = [4, 8, 16, 32, 64, 126.5]  # not 128 for legibility
        # lbls = [str(s) for s in (3, 4, 5, 6, 7, 8)]
        # # max_nbits_left = ax.get_xlim()[1]
        # # if max_nbits_left > 128:
        # #     cutoffs.append(128)
        # for i, cutoff in enumerate(cutoffs):
        #     # ax.plot([cutoff, cutoff], [0, ymax], 'k--', lw=1)
        #     ax.plot([cutoff, cutoff], [0, 1], 'k--', lw=1)
        #     # x = cutoff / float(max_nbits_left)
        #     x = cutoff / 128. + .004
        #     # y = .05 + (.8 * (i % 2)) + (.025 * (i % 4))  # stagger heights
        #     y = .895
        #     # nbits = str(int(np.log2(cutoff) + 1))
        #     nbits = lbls[i]
        #     ax.annotate(nbits, xy=(x, y), xycoords='axes fraction')

    # plot_uresiduals_bits(absDiffs, ax=axes[3])
    # plot_uresiduals_bits(absOffsetDiffs, ax=axes[-1])
    plot_uresiduals_bits(diffs, ax=axes[3])
    plot_uresiduals_bits(diffs_offset, ax=axes[-1])

    _, ymax_left = axes[3].get_ylim()
    _, ymax_right = axes[-1].get_ylim()
    ymax = max(ymax_left, ymax_right)
    [ax.set_ylim([0, ymax]) for ax in (axes[3], axes[-1])]

    plt.tight_layout()


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

    save = True
    # save = False
    small = False

    # numbits = 16
    numbits = 12
    # numbits = 8

    # offset_algo = None
    # offset_algo = 'nn'
    # offset_algo = 'nn7'
    # offset_algo = 'double_delta'  # delta encode deltas
    # offset_algo = '1.5_delta'  # sub half of prev delta from each delta
    offset_algo = 'dyn_delta'  # pick single or double delta for each block
    # offset_algo = ['dyn_delta', 'nn']
    # offset_algo = ['dyn_delta', 'scaled_signs']
    # offset_algo = ['dyn_delta', 'avg']
    # offset_algo = ['dyn_delta', 'mine']
    # offset_algo = ['dyn_delta', 'mine', 'mine']
    # offset_algo = ['dyn_delta', 'mine', 'mine', 'mine']

    num_neighbors = 256
    # num_neighbors = 1024

    neighbor_scales = None
    # neighbor_scales = [(0, 1), (1, 0)]  # amount to left shift, right shift
    # neighbor_scales = [.5, .75, 1.25, 1.5, 1.75, 2]
    # neighbor_scales = [.5, 2]

    # invert_neighbors = False
    invert_neighbors = True

    n = 8

    dsets = ds.smallUCRDatasets() if small else ds.allUCRDatasets()

    # for d in list(dsets)[1:2]:
    for d in dsets:
        plot_dset(d, numbits=numbits, offset_algo=offset_algo,
                  num_neighbors=num_neighbors, scale_factors=neighbor_scales,
                  try_invert=invert_neighbors, n=n)
        if save:
            prefix = "small/" if small else ""
            suffix = ""
            if "nn" in offset_algo:
                suffix = "_nn{}".format(num_neighbors)
                suffix += "_inv{}".format(1 if invert_neighbors else 0)
                if neighbor_scales:
                    # legible_scales = [((2 << s[0]) >> s[1])/2. for s in neighbor_scales]
                    legible_scales = ','.join([str(s) for s in neighbor_scales])
                suffix += "_scale{}".format(legible_scales)

        if n < 50:
            suffix += "_n={}".format(n)

            save_current_plot(d.name, subdir='{}{}b_{}{}'.format(
                prefix, numbits, offset_algo, suffix))

    if not save:
        plt.show()


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    main()
