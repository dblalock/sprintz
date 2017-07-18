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

from . import compress
from . import learning

from .scratch2 import sort_transform


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


def block_nbits_costs(blocks, signed=True):
    """max nbits cost for each row"""
    N, _ = blocks.shape
    blocks = np.abs(blocks) + (blocks >= 0).astype(np.int32)
    maxes = np.max(blocks, axis=1)
    return compress.nbits_cost(maxes, signed=signed).reshape((N, 1))


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

            nn_cost_bits = compress.nbits_cost(costs[nn_idx])
            zeros_cost_bits = compress.nbits_cost(costs[0])
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
        offsetBlocks2[:, i] = (blocks[:, i-1] >> 1) + (blocks[:, i] >> 1)
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


def canal_transform(blocks):
    # note that this is not the exact transform; just an easy approx to see
    # how much it helps
    blocks_out = -np.sort(-blocks, axis=1)  # negate so sort is descending
    # blocks_out = blocks_out[:, ::-1]
    for i in range(7):
        increase = (blocks_out[:, i] - blocks_out[:, i+1]) / (8-i)
        blocks_out[:, i+1:] += increase.reshape((-1, 1))
        blocks_out[:, i] = blocks_out[:, i+1]
    return blocks_out


def inflection_encode(blocks):
    """
    >>> inflection_encode([2, -2, 1, 0])
    array([ 2,  0, -1, -1])
    """
    blocks = np.asarray(blocks)
    shape = blocks.shape
    blocks = blocks.ravel()
    blocks_out = np.copy(blocks)
    for i in range(1, len(blocks)):
        val0 = blocks[i] - blocks[i-1]
        val1 = blocks[i] + blocks[i-1]
        blocks_out[i] = val0 if np.abs(val0) <= np.abs(val1) else val1

    return blocks_out.reshape(shape)


def take_best_of_each(blocks_list, loss='linf', axis=-1, return_counts=False):
    """for each row idx, takes the row from any matrix in blocks_list that has
    the smallest loss (max abs value by default)"""
    best_blocks = np.copy(blocks_list[0])
    # best_costs = np.max(np.abs(best_blocks), axis=1)
    best_costs = learning.compute_loss(best_blocks, loss=loss, axis=axis)
    if return_counts:
        counts = np.zeros(len(blocks_list), dtype=np.int32)
    for i, other_blocks in enumerate(blocks_list[1:]):
        # costs = np.max(np.abs(other_blocks), axis=1)
        costs = learning.compute_loss(other_blocks, loss=loss, axis=axis)
        take_idxs = costs < best_costs
        best_costs = np.minimum(costs, best_costs)
        best_blocks[take_idxs] = other_blocks[take_idxs]

        if return_counts:
            counts[i + 1] = np.sum(take_idxs)

    if return_counts:
        counts[0] = len(best_blocks) - np.sum(counts)
        return best_blocks, counts

    return best_blocks


def compute_deltas(blocks):
    blocks_diff = np.zeros(blocks.size)
    blocks_diff[0] = blocks.ravel()[0]
    blocks_diff[1:] = np.diff(blocks.ravel())
    return blocks_diff.reshape(blocks.shape)


def encode_fir(blocks, filt):
    """
    Note that this includes the prefixes that aren't encoded (eg, first element,
    and first two elements, in below examples)

    >>> encode_fir([0,1,2,2,0,0], [1])  # delta coding
    array([ 0,  1,  1,  0, -2,  0])
    >>> encode_fir([0,1,2,2,0,0], [2, -1])  # double delta coding
    array([ 0,  1,  0, -1, -2,  2])
    """
    # pretty sure this is equivalent to just convolving with [1, -filt]
    filt = np.asarray([0] + list(filt)).ravel()
    ret = np.array(blocks, dtype=filt.dtype)
    shape = ret.shape
    ret = ret.ravel()
    predicted = np.convolve(ret, filt, mode='valid')

    ret[(len(filt)-1):] -= predicted.astype(ret.dtype)
    return ret.reshape(shape)


def possibly_delta_encode(blocks):
    return take_best_of_each([blocks, compute_deltas(blocks)])


def split_dyn_delta(blocks):
    blocks_diff = compute_deltas(blocks)
    blocks_out = np.empty(blocks.shape)
    blocks_out[:, :4] = take_best_of_each([blocks[:, :4], blocks_diff[:, :4]])
    blocks_out[:, 4:] = take_best_of_each([blocks[:, 4:], blocks_diff[:, 4:]])
    return blocks_out


def dyn_filt(blocks, filters=None):
    # filters = ([], [1])  # only delta and delta-delta; should match dyn_delta
    # filters = ([], [1], [2, -1], [.5, .5], [.5, 0, -.5])
    # filters = ([1], [2, -1])  # only delta and delta-delta; should match dyn_delta
    # filters = ([1, 0, 0, 0], [2, -1, 0, 0])  # only delta and delta-delta; should match dyn_delta

    if filters is None:
        # filters = learning.greedy_brute_filters(blocks, nfilters=2)
        filters = learning.greedy_brute_filters(
            # blocks, block_sz=8, nfilters=16, nbits=3, verbose=2)
            blocks, block_sz=8, nfilters=16, nbits=4, verbose=2, step_sz=.25)

        # print "got filters:", filters[:, ::-1]
        # import sys; sys.exit()

        # filters = learning.learn_filters(blocks)[:, ::-1]
        # filters = learning.learn_filters(blocks)[:, ::-1]  # reverse for conv
        # filters = ([0], [1], [2, -1])  # nothing, delta, double delta # TODO rm

        # filters = filters[:, ::-1]  # reverse for conv

    # if True:  # let it cheat and pick 8 filters
    filter_for_each_sample = False
    if filter_for_each_sample:  # let it cheat and pick 8 filters

        ntaps = filters.shape[1]
        x = np.asarray(blocks, dtype=np.float32).ravel()
        X = window.sliding_window_1D(x[:-1], ntaps).astype(np.float32)
        y = x[ntaps:].astype(np.float32).reshape((-1, 1))  # col vect

        predictions = np.dot(X, filters.T)
        raw_errs = y - predictions
        losses = raw_errs * raw_errs

        assigs = np.argmin(losses, axis=1)
        print "    selected counts: ", np.bincount(assigs) / float(len(X))

        ret = np.zeros_like(blocks).ravel()
        all_idxs = np.arange(len(predictions))
        ret[ntaps:] = predictions[all_idxs, assigs]
        ret = blocks.ravel() - ret
        return ret.reshape(blocks.shape)

    blocks_list = [encode_fir(blocks, filt) for filt in filters[:, ::-1]]
    return take_best_of_each(blocks_list, loss='l2').astype(np.int32)
    # blocks, counts = take_best_of_each(
    #     blocks_list, loss='l2', return_counts=True)
    # print "  approx fraction each filter chosen:\n     ", counts / float(np.sum(counts))
    # return blocks.astype(np.int32)


def name_transforms(transforms):
    if not isinstance(transforms, (list, tuple)):
        transforms = [transforms]
    return '|'.join([str(s) for s in transforms])


def apply_transforms(X, diffs, blocks, transform_names, diffs_is_cheating=True):

    if diffs_is_cheating:
        # diffs = encode_fir(X, [1])
        diffs = compute_deltas(X)

    # first round of transforms; defines offsetBlocks var
    if 'delta' in transform_names:  # delta encode deltas
        offsetBlocks = convert_to_blocks(diffs)

    elif 'double_delta' in transform_names:  # delta encode deltas
        # double_diffs = np.copy(diffs)
        # double_diffs[:, 1:] = diffs[:, 1:] - diffs[:, :-1]
        offsetBlocks = convert_to_blocks(compute_deltas(diffs))

    elif 'dyn_delta' in transform_names:  # pick delta or delta-delta
        offsetBlocks = convert_to_blocks(diffs)
        offsetBlocks = possibly_delta_encode(offsetBlocks)

    elif 'split_dyn' in transform_names:
        offsetBlocks = convert_to_blocks(diffs)
        offsetBlocks = split_dyn_delta(offsetBlocks)

    elif 'dyn_filt' in transform_names:
        offsetBlocks = dyn_filt(blocks)

    else:
        print "warning: apply_transforms: using no filter-based transform"
        offsetBlocks = np.copy(blocks)

    if 'center' in transform_names:
        mins = np.min(offsetBlocks, axis=1)
        ranges = np.max(offsetBlocks, axis=1) - mins
        sub_vals = (mins + ranges / 2).astype(np.int32)
        offsetBlocks -= sub_vals.reshape((-1, 1))

    if 'scaled_signs' in transform_names:
        offsetBlocks = scaled_signs_encode(offsetBlocks)

    if 'nn' in transform_names:
        offsetBlocks = nn_encode(offsetBlocks, **nn_kwargs)

    if 'avg' in transform_names:
        offsetBlocks2 = np.empty(offsetBlocks.shape, dtype=np.int32)
        for i in range(1, 8):
            offsetBlocks2[:, i] = offsetBlocks[:, i-1] >> 1 + offsetBlocks[:, i] >> 1
        offsetBlocks2[:, 0] = offsetBlocks[:, 0] - offsetBlocks2[:, 1]
        offsetBlocks = np.copy(offsetBlocks2)

    transform_names_copy = [el for el in transform_names]
    while 'mine' in transform_names_copy:
        # offsetBlocks = my_transform(offsetBlocks)
        offsetBlocks = my_old_transform(offsetBlocks)
        transform_names_copy.remove('mine')
        if 'mine' in transform_names_copy:
            perm = [1, 2, 5, 6, 7, 4, 3, 0]
            offsetBlocks = offsetBlocks[:, perm]

    # if transform_names[0] == 'inflection' :
    if 'inflection' in transform_names:
        offsetBlocks = inflection_encode(offsetBlocks)

    # if transform_names[0] == 'can:
    if 'canal' in transform_names:
        offsetBlocks = canal_transform(offsetBlocks)

    # "examples" stitched together from blocks with vals subbed off
    # if diffs_offset is None:
    use_shape = diffs.shape[0] - 1, diffs.shape[1]
    use_size = use_shape[0] * use_shape[1]
    diffs_offset = offsetBlocks.ravel()[:use_size].reshape(use_shape)

    return offsetBlocks, diffs_offset


def plot_dset(d, numbits=8, left_transforms=None,
              right_transforms=None, prefix_nbits=None, n=100, **nn_kwargs):

    plot_sort = False  # plot distros of idxs into vals sorted by rel freq

    # force transforms to be collections
    if right_transforms is None or isinstance(right_transforms, str):
        right_transforms = (right_transforms,)
    if left_transforms is None or isinstance(left_transforms, str):
        left_transforms = (left_transforms,)

    _, axes = plt.subplots(4, 2, figsize=(10, 8))
    axes = np.concatenate([axes[:, 0], axes[:, 1]])

    # ------------------------ data munging

    # NOTE: blocks used to be blocks of diffs
    # X, diffs, blocks = quantize_diff_block(d.X, numbits, keep_nrows=n)
    X, diffs, _ = quantize_diff_block(d.X, numbits, keep_nrows=n)
    blocks = convert_to_blocks(X)

    offsetBlocksLeft, diffs_left = apply_transforms(
        X, diffs, blocks, left_transforms)
    offsetBlocksRight, diffs_right = apply_transforms(
        X, diffs, blocks, right_transforms)

    # ------------------------ compute where overflows are

    def compute_signed_overflow(resids, nbits):
        cutoff = int(2**(nbits - 1)) - 1
        x_bad = np.where(np.abs(resids).ravel() > cutoff)[0]
        y_bad = resids.ravel()[x_bad]
        x_bad = np.mod(x_bad, resids.shape[1])
        overflow_frac = len(x_bad) / float(resids.size)
        return x_bad, y_bad, cutoff, overflow_frac

    # uncomment if we want offsets for raw diffs again
    # x_bad, y_bad, _, overflow_frac = \
    #     compute_signed_overflow(diffs, numbits)
    # x_bad4, y_bad4, cutoff, overflow_frac4 = \
    #     compute_signed_overflow(diffs, numbits - 4)

    x_bad_left, y_bad_left, _, overflow_frac_left = \
        compute_signed_overflow(diffs_left, numbits)
    x_bad4_left, y_bad4_left, _, overflow_frac4_left = \
        compute_signed_overflow(diffs_left, numbits - 4)

    x_bad_right, y_bad_right, _, overflow_frac_right = \
        compute_signed_overflow(diffs_right, numbits)
    x_bad4_right, y_bad4_right, _, overflow_frac4_right = \
        compute_signed_overflow(diffs_right, numbits - 4)

    # ------------------------ actual plotting

    axes[0].set_title("{} ({} bits)".format(d.name, numbits))

    left_names = name_transforms(left_transforms)
    right_names = name_transforms(right_transforms)

    axes[1].set_title("{} deltas ({:.2f}% overflow)\n{}".format(
        d.name, 100 * overflow_frac_left, left_names))
    axes[-3].set_title("{} deltas ({:.2f}% overflow)\n{}".format(
        d.name, 100 * overflow_frac_right, right_names))

    axes[2].set_title("Deltas ({:.2f}% {}bit overflow)".format(
        100 * overflow_frac4_left, numbits - 4))
    axes[-2].set_title("Deltas ({:.2f}% {}bit overflow)".format(
        100 * overflow_frac4_right, numbits - 4))

    axes[3].set_title("PDF of delta bits, max block bits")
    axes[-1].set_title("PDF of deltas, max block delta after offseting")

    maxval = (1 << numbits) - 1
    axes[0].set_ylim((0, maxval))
    axes[1].set_ylim((-maxval/2, maxval/2))
    cutoff = int(2**(numbits - 1 - 4)) - 1
    axes[2].set_ylim((-cutoff - 1, cutoff))
    axes[-2].set_ylim((-cutoff - 1, cutoff))

    # plot raw data
    plot_examples(X, ax=axes[0])

    # plot transformed deltas
    plot_examples(diffs_left, ax=axes[1])
    axes[1].scatter(x_bad_left, y_bad_left, s=np.pi * 10*10,
                    facecolors='none', edgecolors='r')
    plot_examples(diffs_right, ax=axes[-3])
    axes[-3].scatter(x_bad_right, y_bad_right, s=np.pi * 10*10,
                     facecolors='none', edgecolors='r')
    axes[-3].set_ylim(axes[1].get_ylim())

    # plot how well data fits in a smaller window
    plot_examples(diffs_left, ax=axes[2])
    axes[2].scatter(x_bad4_left, y_bad4_left, s=np.pi * 10*10,
                    facecolors='none', edgecolors='r')
    plot_examples(diffs_right, ax=axes[-2])
    axes[-2].scatter(x_bad4_right, y_bad4_right, s=np.pi * 10*10,
                     facecolors='none', edgecolors='r')

    # # plot numbers of bits taken by each resid
    # axes[-4].set_title("Number of bits required for each sample in each ts")
    # resids_nbits = compress.nbits_cost(diffs_right)
    # img = imshow_better(resids_nbits, ax=axes[-4], cmap='gnuplot2')
    # plt.colorbar(img, ax=axes[-4])
    #
    # alternatively, plot exact distros of residuals
    axes[-4].set_title("(Clipped) distribution of residuals")

    clip_min, clip_max = -129, 128
    nbins = (clip_max - clip_min) / 2
    clipped_resids = np.clip(diffs_right, clip_min, clip_max).ravel()
    sb.distplot(clipped_resids, ax=axes[-4], kde=False, bins=nbins, hist_kws={
                'normed': True, 'range': [clip_min, clip_max]})
    if plot_sort:
        # using prefix_nbits < 10 makes it way worse for 12b stuff cuz vals
        # with the same prefix don't get sorted like they should
        #   -could prolly fix this by pre-sorting assuming decay in prob as
        #   we move away from 0; gets tricky though cuz we offset everything
        #
        # positions = sort_transform(diffs_right, nbits=numbits, prefix_nbits=8)
        positions = sort_transform(diffs_right, nbits=numbits, prefix_nbits=prefix_nbits)
        clipped_positions = np.minimum(positions, clip_max).ravel()
        sb.distplot(clipped_positions, ax=axes[-4], kde=False, bins=nbins, hist_kws={
            'normed': True, 'range': [0, clip_max], 'color': 'g'})


    def plot_uresiduals_bits(resids, ax, plot_sort=plot_sort):
        # have final bar in the histogram contain everything that would
        # overflow 8b
        # max_resid = 128
        # resids = np.minimum(max_resid, resids)
        # bins = [1, 2, 4, 8, 16, 32, 127, max_resid + 1]

        max_nbits = 16
        raw_resids = resids
        resids = compress.nbits_cost(resids).ravel()
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

        if plot_sort:
            positions = sort_transform(raw_resids, nbits=numbits, prefix_nbits=8).ravel()
            position_costs = compress.nbits_cost(positions, signed=False)
            sb.distplot(position_costs, ax=ax, kde=False, bins=bins, hist_kws={
                'normed': True, 'range': [0, max_nbits], 'color': 'g'})

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
        costs = [np.mean(resids), np.mean(block_maxes)]
        styles = ['b--', 'r--']
        if plot_sort:
            costs.append(np.mean(position_costs))
            styles.append('g--')
        for i, (cost, style) in enumerate(zip(costs, styles)):
            # cost = costs[i]
            # style = styles[i]
            ax.plot([cost, cost], [0, 1], style, lw=1)
            x = cost / float(xlim[1])
            # y = .05 + (.8 * (i % 2)) + (.025 * (i % 4))  # stagger heights
            y = .66 + (.12 * (i % 3))  # stagger heights
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
    plot_uresiduals_bits(diffs_left, ax=axes[3])
    plot_uresiduals_bits(diffs_right, ax=axes[-1])

    _, ymax_left = axes[3].get_ylim()
    _, ymax_right = axes[-1].get_ylim()
    ymax = max(ymax_left, ymax_right)
    [ax.set_ylim([0, ymax]) for ax in (axes[3], axes[-1])]

    plt.tight_layout()


def main():
    np.set_printoptions(formatter={'float': lambda x: '{:.3f}'.format(x)})

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

    # ------------------------ experimental params

    # numbits = 16
    numbits = 12
    # numbits = 8

    # left_transforms = None
    # left_transforms = 'delta'
    left_transforms = 'dyn_delta'
    # right_transforms = None
    # right_transforms = 'nn'
    # right_transforms = 'nn7'
    # right_transforms = 'double_delta'  # delta encode deltas
    # right_transforms = '1.5_delta'  # sub half of prev delta from each delta
    # right_transforms = 'dyn_delta'  # pick single or double delta for each block
    right_transforms = 'dyn_filt'
    # right_transforms = ['dyn_delta', 'nn']
    # right_transforms = ['dyn_delta', 'scaled_signs']
    # right_transforms = ['dyn_delta', 'avg']
    # right_transforms = ['dyn_delta', 'mine']
    # right_transforms = ['dyn_delta', 'mine', 'mine']
    # right_transforms = ['dyn_delta', 'mine', 'mine', 'mine']
    # right_transforms = ['dyn_delta', 'inflection']
    # right_transforms = ['dyn_delta', 'canal']
    # right_transforms = ['inflection']
    # right_transforms = 'split_dyn'
    # right_transforms = 'dyn_filt'

    num_neighbors = 256
    # num_neighbors = 1024

    neighbor_scales = None
    # neighbor_scales = [(0, 1), (1, 0)]  # amount to left shift, right shift
    # neighbor_scales = [.5, .75, 1.25, 1.5, 1.75, 2]
    # neighbor_scales = [.5, 2]

    # invert_neighbors = False
    invert_neighbors = True

    save = True
    # save = False
    small = False
    small = True

    # prefix_nbits = 8
    # prefix_nbits = 10
    # prefix_nbits = 11
    prefix_nbits = -1

    n = 8
    # n = 32

    dsets = ds.smallUCRDatasets() if small else ds.allUCRDatasets()

    # ------------------------ name output dir based on params
    suffix = ""
    prefix = "small/" if small else ""
    if "nn" in right_transforms:
        suffix = "_nn{}".format(num_neighbors)
        suffix += "_inv{}".format(1 if invert_neighbors else 0)
        if neighbor_scales:
            legible_scales = ','.join([str(s) for s in neighbor_scales])
        suffix += "_scale{}".format(legible_scales)
    if n < 50:
        suffix += "_n={}".format(n)
    if prefix_nbits > 0:
        suffix += '_prefix={}b'.format(prefix_nbits)
    transforms_str = "{}-vs-{}".format(name_transforms(left_transforms),
                                       name_transforms(right_transforms))
    subdir = '{}{}b_{}{}'.format(prefix, numbits, transforms_str, suffix)
    print "saving to subdir: {}".format(subdir)

    # ------------------------ main loop

    # for d in list(dsets)[26:27]:  # olive oil
    # for d in list(dsets)[1:2]:  # adiac
    # for i, d in enumerate(dsets):
    # for d in dsets:
    # for d in list(dsets)[4:8]:
    # for d in list(dsets)[:4]:
    for d in list(dsets)[:16]:
        print "------------------------ {}".format(d.name)
        plot_dset(d, numbits=numbits, n=n,
                  left_transforms=left_transforms,
                  right_transforms=right_transforms,
                  num_neighbors=num_neighbors,
                  scale_factors=neighbor_scales,
                  try_invert=invert_neighbors,
                  prefix_nbits=prefix_nbits)
        if save:
            save_current_plot(d.name, subdir=subdir)

    if not save:
        plt.show()


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    main()
