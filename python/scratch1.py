#!/usr/bin/env python

import itertools
import os
# import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from scipy.ndimage import filters

# from .datasets import ucr
# from . import datasets as ds
from python.datasets import ucr
from python.utils import files
from python.utils import sliding_window as window

from python import compress
from python import hashing
from python import learning
from python import learning2

from .scratch2 import sort_transform, mixfix_encode, mixfix_cost, zigzag_encode
from .scratch2 import prefix_lut_transform

from python.datasets import compress_bench  # for some preproc

FIG_SAVE_DIR = 'figs/'
# SAVE_DIR = 'figs/ucr/'


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
    saveDir = os.path.join(os.path.expanduser(FIG_SAVE_DIR), subdir)
    files.ensure_dir_exists(saveDir)
    fileName = os.path.join(saveDir, datasetName) + suffix
    plt.savefig(fileName)
    plt.close()


def filter_rows(X, filt_len, kind='hamming', scale_filter_how='sum1'):
    if kind == 'hamming':
        filt = np.hamming(filt_len)
    elif kind == 'flat':
        filt = np.ones(filt_len)
    else:
        raise RuntimeError("Unknown/unsupported filter type {}".format(kind))
    if scale_filter_how == 'max1':
        filt /= np.max(filt)
    elif scale_filter_how == 'sum1':
        filt /= np.sum(filt)

    return filters.convolve1d(X, weights=filt, axis=1, mode='constant')


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

    # plot best fit power law
    # EDIT: moral of this story is that it's not really a power law either;
    # if you take log of x, it's more like a gaussian around 0
    def fit_power_law(xvals, yvals):
        logx = np.log(xvals)
        logy = np.log(yvals)

        ignore_idxs = np.isinf(logx)  # shouldn't happen, but why not
        ignore_idxs += np.isinf(logy)  # happens for 0 counts
        keep_idxs = ~ignore_idxs
        logx, logy = logx[keep_idxs], logy[keep_idxs]

        # print "nans in logx: ", np.where(np.isnan(logx))[0]
        # print "nans in logy: ", np.where(np.isnan(logy))[0]
        # print "infs in logx: ", np.where(np.isinf(logx))[0]
        # print "infs in logy: ", np.where(np.isinf(logy))[0]

        # from scipy import optimize
        # fitfunc = lambda p, x: p[0] + p[1] * x  # noqa
        # errfunc = lambda p, x, y: y - fitfunc(p, x)  # noqa
        # p_init = [1.0, -1.0]
        # p, _ = optimize.leastsq(errfunc, p_init, args=(logx, logy))

        # from scipy import polyfit, polyval
        from scipy.stats import linregress
        slope, intercept, corr_r, _, _ = linregress(logx, logy)
        print "power law slope, intercept, corr = {}, {}, {}".format(
            slope, intercept, corr_r)

        # yhat = np.exp(-intercept + slope * np.log(xvals))
        yhat = np.exp(intercept + slope * np.log(xvals))
        # log_yhat = p[0] + p[1] * logx
        # yhat = np.exp(log_yhat)

        # plt.figure()
        # # plt.plot(xvals, yvals)
        # # plt.plot(xvals, yhat)
        # plt.plot(logx, logy)
        # plt.plot(np.log(xvals), np.log(yhat))
        # plt.show()

        return yhat

    # # print "min clipped resid, clip_min: ", np.min(clipped_resids), clip_min
    # counts = np.bincount(clipped_resids - clip_min)[1:-1]  # final bins have whole tails
    # # x_neg = np.arange(clip_min + 1, 1)  # duh; no log of negative stuff
    # # yhat_neg = fit_power_law(x_neg, counts[:len(x_neg)])
    # # axes[-4].plot(x_neg, yhat_neg)
    # x_pos = np.arange(0, clip_max)
    # yhat_pos = fit_power_law(x_pos, counts[-len(x_pos):])
    # # axes[-4].plot(x_pos, np.log(yhat_pos))
    # axes[-4].plot(x_pos, yhat_pos)

    def cauchy_pdf(x, g, x_mean=0):
        dist = (x - x_mean) / g
        denom = np.pi * g * (1. + dist*dist)
        return 1. / denom

    def log_cauchy_pdf(x, g, x_mean=0):
        dist = (x - x_mean) / g
        return -np.log(np.pi * g) - np.log(1 + dist*dist)

    def fit_cauchy(xvals, yvals, x_mean=0):
        # from scipy import optimize
        # # see https://en.wikipedia.org/wiki/Cauchy_distribution
        # fitfunc = lambda g, x: cauchy_pdf(x, g, x_mean=x_mean) # noqa
        # errfunc = lambda g, x, y: y - fitfunc(g, x)  # noqa
        # gamma = np.std(yvals)
        # # gamma, _ = optimize.leastsq(errfunc, gamma, args=(xvals, yvals))

        # yhat = cauchy_pdf(xvals, gamma, x_mean=x_mean)
        # return yhat

        from scipy import stats
        loc, scale = stats.cauchy.fit(yvals)
        # loc, scale = stats.cauchy.fit(np.log(yvals))

        # print "model: ", model
        # import sys; sys.exit()
        loc = 0  # hardcode mean of 0
        yhat = stats.cauchy.pdf(xvals, loc=loc, scale=scale)
        return yhat

    def fit_lomax(xvals, yvals):
        # double_abs_vals = 2 * np.abs(yvals)  # since would be zigzag encoded
        double_abs_vals = np.abs(yvals)  # since would be zigzag encoded
        lamda0 = np.mean(double_abs_vals)
        # alpha0 = 1
        alpha0 = lamda0 * lamda0  # set mean of gamma to true lamda
        alpha = alpha0 + len(yvals)
        lamda = lamda0 + np.sum(double_abs_vals)

        def lomax_pdf(x, alpha, lamda):
            return alpha / lamda * np.power(1. + (x / lamda), -(alpha + 1))

        # mul by 2 cuz
        yhat = lomax_pdf(2 * np.abs(xvals), alpha, lamda)
        return yhat


DEFAULT_MAX_NN_IDX = (1 << 11) - 1


def block_nbits_costs(blocks, signed=True):
    """max nbits cost for each row"""
    N, _ = blocks.shape
    blocks = np.abs(blocks) + (blocks >= 0).astype(np.int32)
    maxes = np.max(blocks, axis=1)
    return compress.nbits_cost(maxes, signed=signed).reshape((N, 1))


# def nn_encode(diffs, num_neighbors, nn_step=4, nn_hash=None):
def nn_encode(blocks, num_neighbors=256, nn_step=1, nn_hash=None,
              scale_factors=None, try_invert=True, nn_loss='l2',
              mean_normalize=False, nn_nblocks=1, predict_next=False, **sink):

    orig_shape = blocks.shape
    if nn_nblocks > 1:
        blocks = learning.trim_to_multiple_of(blocks, nn_nblocks)  # TODO move to utils
        orig_shape = blocks.shape
        new_shape = blocks.shape[0] / nn_nblocks, blocks.shape[1] * nn_nblocks
        blocks = blocks.reshape(new_shape)

    if scale_factors is not None and 1 in scale_factors:
        scale_factors = scale_factors[:]
        scale_factors.remove(1)

    if nn_hash is None or nn_hash == 'none':
        # assert nn_step == 4  # we only handle 4 for now
        # diff_windows = window.sliding_window_1D(diffs.ravel(), 8, step=8)
        # offsetBlocks = np.zeros_like(diff_windows)
        offsetBlocks = np.copy(blocks)

        # N = len(diff_windows)
        N, M = blocks.shape
        # zero_samples = np.zeros(M, dtype=np.int32)

        end = N - 1 if predict_next else N
        nn_idxs = np.zeros(end - 1, dtype=np.int)
        times_nonzero = 0
        saved_bits = 0
        saved_bits_sq = 0
        for n in range(1, end):
            # start_idx = int(max(0, n - (nn_step / 8.) * num_neighbors))
            # end_idx = n
            # history = diff_windows[start_idx:end_idx].ravel()
            # nn_windows = window.sliding_window_1D(history, 8, step=4)
            # neighbors = np.vstack((nn_windows, zero_samples))

            # this isn't exactly where we'd start in the flattened array,
            # but whatever
            start_idx = n - 1 - ((num_neighbors - 1) * nn_step / M)
            start_idx = max(0, start_idx)
            end_idx = n
            history = blocks[start_idx:end_idx].ravel()

            nn_windows = window.sliding_window_1D(history, M, step=nn_step)
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

            # if not predict_next:
            #     all_neighbors = [zero_samples] + all_neighbors
            neighbors = np.vstack(all_neighbors)

            query = np.copy(blocks[n])
            query_mean = 0
            if mean_normalize:
                neighbors -= np.mean(neighbors, axis=1, keepdims=True).astype(np.int32)
                query_mean = np.mean(query).astype(np.int32)
                query -= query_mean

            # costs = np.abs(neighbors - blocks[n])
            # costs = np.max(costs, axis=1)
            errs = neighbors - query
            costs = learning.compute_loss(errs, loss=nn_loss)
            nn_idx = np.argmin(costs)

            scale_neighbor_by = 1
            # scale_neighbor_by = .5  # TODO make param

            if nn_idx == len(neighbors) - 1:
                continue  # hack to avoid edge case in next line
            neighbor = neighbors[nn_idx + 1] if predict_next else neighbors[nn_idx]
            target = blocks[n + 1] if predict_next else blocks[n]
            target_mean = np.mean(target) if mean_normalize else 0
            resids = target - (neighbor * scale_neighbor_by + target_mean)

            orig_cost = learning.compute_loss(target, loss=nn_loss)
            nn_cost = learning.compute_loss(resids, loss=nn_loss)

            if nn_cost < orig_cost:
                times_nonzero += 1
                write_block = resids
                nn_idxs[n-1] = nn_idx
            else:
                write_block = target
                nn_idxs[n-1] = 0

            # if nn_loss == 'linf':
                # if predict_next:
                #     neighbor = neighbors[nn_idx]
                #     nn_cost_bits = compress.nbits_cost(blocks[n+1])
                #     zeros_cost_bits = compress.nbits_cost(blocks[n+1])
                # else:
                #     nn_cost_bits = compress.nbits_cost(costs[nn_idx])
                #     zeros_cost_bits = compress.nbits_cost(costs[0])
                # bitsave = zeros_cost_bits - nn_cost_bits
                # bitsave = max(0, orig_cost - nn_cost)
            orig_nbits = np.max(compress.nbits_cost(target))
            new_nbits = np.max(compress.nbits_cost(resids))
            bitsave = max(0, orig_nbits - new_nbits)
            saved_bits += bitsave
            saved_bits_sq += bitsave*bitsave

            write_idx = n + 1 if predict_next else n
            offsetBlocks[write_idx] = write_block

            # times_nonzero += int(nn_idx > 0)

        # if nn_loss == 'linf':
        N_f = float(N)
        expected_bitsave = saved_bits / N_f
        std_bitsave = saved_bits_sq / N_f - expected_bitsave * expected_bitsave
        print "nn nonzero {}/{} ({:.1f}%) and saved {:.2f} +/-{:.2f} bits" \
            .format(times_nonzero, N, 100 * times_nonzero / N_f,
                    expected_bitsave, std_bitsave)

        print "mean, std of nn idxs: ", np.mean(nn_idxs), np.std(nn_idxs)

        return offsetBlocks.reshape(orig_shape)

        # TODO option to use various hash funcs here

    raise ValueError("Unrecognized nn_hash: '{}'".format(nn_hash))


def convert_to_blocks(diffs):
    tail = diffs.size % 8
    blocks = diffs.ravel()[:-tail] if tail > 0 else diffs.ravel()
    return blocks.reshape((-1, 8))


def quantize(X, numbits, keep_nrows=-1, mean_norm=False, stitch_ends=False):
    if keep_nrows > 0 and keep_nrows < len(X):
        np.random.seed(123)
        which_idxs = np.random.choice(len(X), keep_nrows, replace=False)
        X = X[which_idxs]

    if mean_norm:
        X -= np.mean(X, axis=1, keepdims=True)

    if stitch_ends:
        for i in range(1, len(X)):
            jump = X[i, 0] - X[i-1, -1]
            X[i] -= jump

    maxval = (1 << numbits) - 1
    np.random.seed(123)
    X = X - np.min(X)

    print "X shape: ", X.shape
    print "X min, max: ", X.min(), X.max()
    dtype = np.int32 if numbits <= 32 else np.int64
    X = (maxval / float(np.max(X)) * X).astype(dtype)
    print "quantized X min, max: ", X.min(), X.max()
    # print "initial quantized data: "
    # print X.ravel()[:25]

    # diffs = X[:, 1:] - X[:, :-1]
    # blocks = convert_to_blocks(diffs)
    # return X, diffs, blocks

    return X


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


# def split_dyn_delta(blocks):
#     blocks_diff = delta_encode(blocks)
#     blocks_out = np.empty(blocks.shape)
#     blocks_out[:, :4] = take_best_of_each([blocks[:, :4], blocks_diff[:, :4]])
#     blocks_out[:, 4:] = take_best_of_each([blocks[:, 4:], blocks_diff[:, 4:]])
#     return blocks_out


# def split_dyn_filt(blocks):
#     # blocks_out = np.empty(blocks.shape)
#     # filters = learning.greedy_brute_filters(
#         # blocks, block_sz=4, nfilters=4, nbits=4, verbose=2, step_sz=.25,
#         # blocks, block_sz=4, nfilters=4, nbits=3, verbose=2, step_sz=.5,
#         # loss='l2')
#     shape = blocks.shape
#     assert shape[1] == 8
#     new_shape = (shape[0] * 2, shape[1] / 2)
#     blocks_out = dyn_filt(blocks.reshape(new_shape), block_sz=4, nfilters=4)
#     # blocks_out[:, :4] = dyn_filt(blocks[:, :4], filters=filters)
#     # blocks_out[:, 4:] = dyn_filt(blocks[:, 4:], filters=filters)
#     return blocks_out.reshape(shape)


def take_best_of_each(blocks_list, loss='linf', axis=-1, return_counts=False):
    """for each row idx, takes the row from any matrix in blocks_list that has
    the smallest loss (max abs value by default)"""
    best_blocks = np.copy(blocks_list[0])

    all_costs = [learning.compute_loss(blocks, loss=loss, axis=axis) for blocks in blocks_list]
    best_costs = all_costs[0]
    # best_costs = learning.compute_loss(best_blocks, loss=loss, axis=axis)
    # if return_counts:
        # counts = np.zeros(len(blocks_list), dtype=np.int32)
    for i, other_blocks in enumerate(blocks_list[1:]):
        # costs = np.max(np.abs(other_blocks), axis=1)
        # costs = learning.compute_loss(other_blocks, loss=loss, axis=axis)
        costs = all_costs[i + 1]
        take_idxs = costs < best_costs
        best_costs = np.minimum(costs, best_costs)
        best_blocks[take_idxs] = other_blocks[take_idxs]

        # if return_counts:
        #     counts[i + 1] = np.sum(take_idxs)

    if return_counts:
        all_costs = np.vstack(all_costs)
        argmins = np.argmin(all_costs, axis=0)
        counts = np.bincount(argmins)
        return best_blocks, counts
        # counts[0] = len(best_blocks) - np.sum(counts)
        # return best_blocks, counts

    return best_blocks


def delta_encode(blocks):
    blocks_diff = np.zeros(blocks.size)
    blocks_diff[0] = blocks.ravel()[0]
    blocks_diff[1:] = np.diff(blocks.ravel())
    return blocks_diff.reshape(blocks.shape)


def dyn_delta_encode(blocks):
    offsetBlocks = delta_encode(blocks)
    return maybe_delta_encode(offsetBlocks)


def encode_fir(blocks, filt):
    # """
    # Note that this includes the prefixes that aren't encoded (eg, first element,
    # and first two elements, in below examples)

    # >>> encode_fir([0,1,2,2,0,0], [1])  # delta coding
    # array([ 0,  1,  1,  0, -2,  0])
    # >>> encode_fir([0,1,2,2,0,0], [2, -1])  # double delta coding
    # array([ 0,  1,  0, -1, -2,  2])
    # >>> encode_fir([0,1,2,2,0,0], [1, 0])  # delta coding, longer filt
    # array([ 0,  1,  1,  0, -2,  0])
    # """
    # pretty sure this is equivalent to just convolving with [1, -filt]
    # print "convolving with filter: ", filt

    filt = np.asarray([0] + list(filt)).ravel()

    ret = np.array(blocks, dtype=filt.dtype)
    shape = ret.shape
    ret = ret.ravel()
    predicted = np.convolve(ret, filt, mode='valid')

    ret[(len(filt)-1):] -= predicted.astype(ret.dtype)
    return ret.reshape(shape)


def maybe_delta_encode(blocks):
    ret, counts = take_best_of_each([blocks, delta_encode(blocks)], return_counts=True)
    print "delta, delta-delta fracs: ", counts / float(counts.sum())
    return ret


def dyn_filt(blocks, filters=None, **learn_kwargs):
    # filters = ([], [1])  # only delta and delta-delta; should match dyn_delta
    # filters = ([], [1], [2, -1], [.5, .5], [.5, 0, -.5])
    # filters = ([1], [2, -1])  # only delta and delta-delta; should match dyn_delta
    # filters = ([1, 0, 0, 0], [2, -1, 0, 0])  # only delta and delta-delta; should match dyn_delta

    # print "initial blocks shape: ", blocks.shape

    if filters is None:
        # filters = learning.greedy_brute_filters(blocks, nfilters=2)
        learn_kwargs.setdefault('block_sz', blocks.shape[1])
        # learn_kwargs.setdefault('nfilters', 16)
        # learn_kwargs.setdefault('nfilters', 4)
        learn_kwargs.setdefault('nfilters', 1)
        learn_kwargs.setdefault('nbits', 3)
        learn_kwargs.setdefault('ntaps', 3)
        learn_kwargs.setdefault('verbose', 2)
        learn_kwargs.setdefault('step_sz', .5)
        # learn_kwargs.setdefault('loss', 'linf')
        learn_kwargs.setdefault('loss', 'l2')
        filters = learning.greedy_brute_filters(blocks, **learn_kwargs)

        # filters = filters[:2]  # TODO rm after debug
        # filters[1, :] = 0

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

    # print "blocks shape: ", blocks.shape

    blocks_list = [encode_fir(blocks, filt) for filt in filters[:, ::-1]]
    # return take_best_of_each(blocks_list, loss='l2').astype(np.int32)
    # return take_best_of_each(blocks_list, loss='linf').astype(np.int32)

    blocks, counts = take_best_of_each(
        blocks_list, loss='linf', return_counts=True)
    print "  fraction each filter chosen:\n     ", counts / float(np.sum(counts))

    # print "final blocks shape: ", blocks.shape

    return blocks.astype(np.int32)


def name_transforms(transforms):
    if not isinstance(transforms, (list, tuple)):
        transforms = [transforms]
    return '|'.join([str(s) for s in transforms])


def apply_transforms(X, blocks, transform_names, k=-1, side=None,
                     chunk_sz=-1, numbits=8, **kwargs):
    # diffs_is_cheating=True, k=-1, **nn_kwargs):

    # if diffs_is_cheating:
    #     # diffs = encode_fir(X, [1])
        # diffs = delta_encode(X)

    k_left = kwargs.get('k_left', 0)
    k_right = kwargs.get('k_right', 0)
    if side == 'left' and k_left > 0:
        k = k_left
    elif side == 'right' and k_right > 0:
        k = k_right

    transform_names = transform_names[:]

    offsetBlocks = np.copy(blocks)

    while len(transform_names):
        name = transform_names[0]
        if name is None:
            transform_names = transform_names[1:]  # pop first transform
            continue
        if name is 'sub_mean':
            offsetBlocks = offsetBlocks - np.mean(X)

        # first round of transforms; defines offsetBlocks var
        if name == 'delta':  # delta encode deltas
            offsetBlocks = delta_encode(offsetBlocks)

        elif name == 'double_delta':  # delta encode deltas
            offsetBlocks = delta_encode(delta_encode(offsetBlocks))

        elif name == 'dyn_delta':  # pick delta or delta-delta
            offsetBlocks = dyn_delta_encode(offsetBlocks)
            # offsetBlocks = delta_encode(offsetBlocks)
            # offsetBlocks = _delta_encode(offsetBlocks)

        elif name == 'maybe_delta':
            offsetBlocks = maybe_delta_encode(offsetBlocks)

        # elif name == 'split_dyn':
        #     offsetBlocks = delta_encode(offsetBlocks)
        #     offsetBlocks = split_dyn_delta(offsetBlocks)

        elif name == 'dyn_filt':
            offsetBlocks = dyn_filt(offsetBlocks)

        # elif name == 'split_dyn_filt':
        #     offsetBlocks = split_dyn_filt(blocks)

        if name == 'center':
            mins = np.min(offsetBlocks, axis=1)
            ranges = np.max(offsetBlocks, axis=1) - mins
            sub_vals = (mins + ranges / 2).astype(np.int32)
            offsetBlocks -= sub_vals.reshape((-1, 1))

        if name == 'scaled_signs':
            offsetBlocks = scaled_signs_encode(offsetBlocks)

        if name == 'nn':
            print "actually doing nn encoding"
            offsetBlocks = nn_encode(offsetBlocks, **kwargs)

        if name == 'avg':
            offsetBlocks2 = np.empty(offsetBlocks.shape, dtype=np.int32)
            for i in range(1, 8):
                offsetBlocks2[:, i] = offsetBlocks[:, i-1] >> 1 + offsetBlocks[:, i] >> 1
            offsetBlocks2[:, 0] = offsetBlocks[:, 0] - offsetBlocks2[:, 1]
            offsetBlocks = np.copy(offsetBlocks2)

        while name == 'mine':
            # offsetBlocks = my_transform(offsetBlocks)
            offsetBlocks = my_old_transform(offsetBlocks)
            # transform_names_copy.remove('mine')
            # if 'mine' in transform_names_copy:
            if 'mine' in transform_names:
                perm = [1, 2, 5, 6, 7, 4, 3, 0]
                offsetBlocks = offsetBlocks[:, perm]

        if name == 'inflection':
            offsetBlocks = inflection_encode(offsetBlocks)

        if name == 'canal':
            offsetBlocks = canal_transform(offsetBlocks)

        if name == 'kmeans':
            offsetBlocks = learning.sub_kmeans(offsetBlocks, k=k)

        if name == 'online_kmeans':
            offsetBlocks = learning.sub_online_kmeans(offsetBlocks, k=k)

        if name == 'VAR':
            offsetBlocks = learning.var_transform(offsetBlocks, chunk_sz=chunk_sz)

        if name.startswith('blocklen'):
            blocklen = int(name.split('=')[1])
            nblocks = int(offsetBlocks.size / blocklen)
            newsize = nblocks * blocklen
            # print "blocklen, newsize = ", blocklen, newsize
            offsetBlocks = offsetBlocks.ravel()[:newsize].reshape(-1, blocklen)

        if name == 'autocoracle':  # autocorr using length of X
            x = offsetBlocks.ravel()
            lag = X.shape[1]
            # print "offsetBlocks shape, size: ", offsetBlocks.shape, offsetBlocks.size
            # print "<autocoracle> X shape, size: ", X.shape, X.size
            corr_mat = np.corrcoef(x[lag:], x[:-lag])
            print "autocorr at true lag: ", corr_mat[0, 1]  # off diag
            x[lag:] = x[lag:] - x[:-lag]
            offsetBlocks = x.reshape(offsetBlocks.shape)

        if name == 'hash':
            offsetBlocks = hashing.hash_predict_transform(offsetBlocks)

        if name == 'prefix_lut':
            offsetBlocks = prefix_lut_transform(offsetBlocks)

        if name == 'sort_transform':
            offsetBlocks = sort_transform(offsetBlocks)

        if name == 'online_regress':
            offsetBlocks = learning2.sub_online_regress(offsetBlocks, **kwargs)

        if name == 'global_regress':
            offsetBlocks = learning2.sub_online_regress(
                offsetBlocks, group_sz_blocks=-1, numbits=numbits, **kwargs)

        if name == 'OnlineGradDescent':
            offsetBlocks = learning2.sub_online_regress(
                offsetBlocks, method='gradient', numbits=numbits, **kwargs)

        if name == 'online_linreg':
            offsetBlocks = learning2.sub_online_regress(
                offsetBlocks, method='exact', numbits=numbits, **kwargs)

        # if name == 'downsample2':  # breaks everything
        #     offsetBlocks = offsetBlocks[:, ::2]

        transform_names = transform_names[1:]  # pop first transform

    # "examples" stitched together from blocks with vals subbed off
    # if diffs_offset is None:
    use_shape = X.shape[0] - 1, X.shape[1]
    use_size = use_shape[0] * use_shape[1]
    diffs_offset = offsetBlocks.ravel()[:use_size].reshape(use_shape)

    # # show autocorrelatations in bit costs
    # all_costs = compress.nbits_cost(offsetBlocks.ravel()).astype(np.float32)
    # all_costs -= np.mean(all_costs)
    # norm = np.mean(all_costs * all_costs)
    # corrs = np.empty(15)
    # for i in range(1, len(corrs) + 1):
    #     corrs[i-1] = np.mean(all_costs[i:] * all_costs[:-i]) / norm
    # print "nbits autocorrelations: ", corrs

    return offsetBlocks.astype(np.int32), diffs_offset.astype(np.int32)


def plot_dset(d, numbits=8, n=100, left_transforms=None, right_transforms=None,
              prefix_nbits=None, **transform_kwargs):

    plot_sort = False  # plot distros of idxs into vals sorted by rel freq
    plot_mixfix = False  # plot distros of codes after mixfix encoding
    plot_sub_minbits = False  # plot distros of vals above min nbits value

    # force transforms to be collections
    if right_transforms is None or isinstance(right_transforms, str):
        right_transforms = (right_transforms,)
    if left_transforms is None or isinstance(left_transforms, str):
        left_transforms = (left_transforms,)

    fig, axes = plt.subplots(4, 2, figsize=(10, 8))
    axes = np.concatenate([axes[:, 0], axes[:, 1]])

    # ------------------------ data munging

    # NOTE: blocks used to be blocks of diffs
    # X, diffs, blocks = quantize(d.X, numbits, keep_nrows=n)
    X = quantize(d.X, numbits, keep_nrows=n)
    X_left = X
    X_right = X
    while left_transforms[0] == 'smooth':
        X_left = filter_rows(X_left, 8)
        left_transforms = left_transforms[1:]
    while right_transforms[0] == 'smooth':
        X_right = filter_rows(X_right, 8)
        right_transforms = right_transforms[1:]

    blocks_left = convert_to_blocks(X_left)
    blocks_right = convert_to_blocks(X_right)

    transform_kwargs.update({'numbits': numbits})
    offsetBlocksLeft, errs_left = apply_transforms(
        X_left, blocks_left, left_transforms, side='left', **transform_kwargs)
    offsetBlocksRight, errs_right = apply_transforms(
        X_right, blocks_right, right_transforms, side='right', **transform_kwargs)

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
        compute_signed_overflow(errs_left, numbits)
    x_bad4_left, y_bad4_left, _, overflow_frac4_left = \
        compute_signed_overflow(errs_left, numbits - 4)

    x_bad_right, y_bad_right, _, overflow_frac_right = \
        compute_signed_overflow(errs_right, numbits)
    x_bad4_right, y_bad4_right, _, overflow_frac4_right = \
        compute_signed_overflow(errs_right, numbits - 4)

    # ------------------------ actual plotting

    axes[0].set_title("{} ({} bits)".format(d.name, numbits))

    left_names = name_transforms(left_transforms)
    right_names = name_transforms(right_transforms)

    axes[1].set_title("{} errors ({:.2f}% overflow)\nAlgorithm = '{}'".format(
        d.name, 100 * overflow_frac_left, left_names))
    axes[-3].set_title("{} errors ({:.2f}% overflow)\nAlgorithm = '{}'".format(
        d.name, 100 * overflow_frac_right, right_names))

    axes[2].set_title("Errors ({:.2f}% need >{}bits)".format(
        100 * overflow_frac4_left, numbits - 4))
    axes[-2].set_title("Errors ({:.2f}% need >{} bits)".format(
        100 * overflow_frac4_right, numbits - 4))

    axes[3].set_title("PDF of inidividual #bits (blue),\nmax #bits in block (red)")
    axes[-1].set_title("PDF of individual #bits (blue),\nmax #bits in block (red)")

    maxval = (1 << numbits) - 1
    axes[0].set_ylim((0, maxval))
    axes[1].set_ylim((-maxval/2, maxval/2))
    cutoff = int(2**(numbits - 1 - 4)) - 1
    axes[2].set_ylim((-cutoff - 1, cutoff))
    axes[-2].set_ylim((-cutoff - 1, cutoff))

    # plot raw data
    plot_examples(X, ax=axes[0])

    # plot transformed deltas
    plot_examples(errs_left, ax=axes[1])
    axes[1].scatter(x_bad_left, y_bad_left, s=np.pi * 10*10,
                    facecolors='none', edgecolors='r')
    plot_examples(errs_right, ax=axes[-3])
    axes[-3].scatter(x_bad_right, y_bad_right, s=np.pi * 10*10,
                     facecolors='none', edgecolors='r')
    axes[-3].set_ylim(axes[1].get_ylim())

    # plot how well data fits in a smaller window
    plot_examples(errs_left, ax=axes[2])
    axes[2].scatter(x_bad4_left, y_bad4_left, s=np.pi * 10*10,
                    facecolors='none', edgecolors='r')
    plot_examples(errs_right, ax=axes[-2])
    axes[-2].scatter(x_bad4_right, y_bad4_right, s=np.pi * 10*10,
                     facecolors='none', edgecolors='r')

    # ------------------------ residuals distro in top right

    # # plot numbers of bits taken by each resid
    # axes[-4].set_title("Number of bits required for each sample in each ts")
    # resids_nbits = compress.nbits_cost(errs_right)
    # img = imshow_better(resids_nbits, ax=axes[-4], cmap='gnuplot2')
    # plt.colorbar(img, ax=axes[-4])

    axes[-4].set_title("(Clipped) distribution of residuals")

    clip_min, clip_max = -129, 128
    nbins = (clip_max - clip_min) / 2
    clipped_resids = np.clip(errs_right, clip_min, clip_max).astype(np.int32).ravel()
    # clipped_resids = np.log(clipped_resids) # TODO rm
    sb.distplot(clipped_resids, ax=axes[-4], kde=False, bins=nbins, hist_kws={
                'normed': True, 'range': [clip_min, clip_max]})
    if plot_sort:
        # using prefix_nbits < 10 makes it way worse for 12b stuff cuz vals
        # with the same prefix don't get sorted like they should
        #   -could prolly fix this by pre-sorting assuming decay in prob as
        #   we move away from 0; gets tricky though cuz we offset everything
        #
        # positions = sort_transform(errs_right, nbits=numbits, prefix_nbits=8)
        positions = sort_transform(errs_right, nbits=numbits, prefix_nbits=prefix_nbits)
        clipped_positions = np.minimum(positions, clip_max).ravel()
        sb.distplot(clipped_positions, ax=axes[-4], kde=False, bins=nbins, hist_kws={
            'normed': True, 'range': [0, clip_max], 'color': 'g'})

    # TODO uncomment
    # if plot_mixfix:
    #     codes = mixfix_encode(errs_right, nbits=numbits)
    #     clipped_codes = np.minimum(codes, clip_max).ravel()
    #     sb.distplot(clipped_codes, ax=axes[-4], kde=False, bins=nbins, hist_kws={
    #         'normed': True, 'range': [0, clip_max], 'color': 'g'})

    # if plot_sub_minbits:
    #     use_blocks = zigzag_encode(offsetBlocksRight)
    #     sub_minbits_block_costs = compress.nbits_cost(use_blocks)
    #     sub_minbits_min_costs = np.min(sub_minbits_block_costs, axis=1, keepdims=True)
    #     sub_minbits_max_costs = np.max(sub_minbits_block_costs, axis=1, keepdims=True)
    #     offset_vals = use_blocks - (1 << sub_minbits_min_costs)
    #     offset_vals /= 2  # wrong, but allows apples-to-apples viz comparison
    #     clipped_vals = np.clip(offset_vals, 0, clip_max).astype(np.int32).ravel()
    #     sb.distplot(clipped_vals, ax=axes[-4], kde=False, bins=nbins,
    #                 hist_kws={'normed': True, 'range': [0, clip_max],
    #                           'color': 'grey'})

    # # plot best fit laplace distro
    # rate = 1. / np.mean(np.abs(errs_right))
    # xvals = np.arange(clip_min, clip_max + 1)
    # yvals = .5 * rate * np.exp(-rate * np.abs(xvals))
    # axes[-4].plot(xvals, yvals, 'k--', lw=1)
    # # axes[-4].plot(xvals, np.log(yvals), 'k--') # TODO rm

    # xvals = np.arange(clip_min + 1, clip_max)
    # counts = np.bincount(clipped_resids - clip_min)[1:-1]  # final bins have whole tails
    # yvals_exp = yvals
    # # fit cauchy
    # yhat_cauchy = fit_cauchy(xvals, counts)
    # axes[-4].plot(xvals, yhat_cauchy, 'c--', lw=1.5)
    # axes[-4].plot(xvals, np.exp(yhat), 'c--', lw=1.5)

    # # fit lomax
    # yhat_lomax = fit_lomax(xvals, counts)
    # axes[-4].plot(xvals, yhat_lomax, 'r--', lw=1.5)

    axes[-4].set_yscale('log')

    # # now compute xent in bits between true distro and approx distros
    # probs = counts / np.sum(counts).astype(np.float32)
    # entropy = -np.nansum(probs * np.log2(probs))
    # xent_exp = -np.sum(probs * np.log2(yvals_exp[1:-1]))
    # # xent_lomax = -np.sum(probs * np.log2(yhat_lomax))
    # xent_cauchy = -np.sum(probs * np.log2(yhat_cauchy))
    # print "entropy, exp xent, lomax xent, cauchy xent: {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(
    #     entropy, xent_exp, xent_lomax, xent_cauchy)

    # ------------------------ histograms of residuals in bottom axes

    def plot_uresiduals_bits(resid_blocks, ax, plot_sort=plot_sort):
        # have final bar in the histogram contain everything that would
        # overflow 8b
        # max_resid = 128
        # resids = np.minimum(max_resid, resids)
        # bins = [1, 2, 4, 8, 16, 32, 127, max_resid + 1]

        max_nbits = 16
        bins = np.arange(max_nbits + 1)

        raw_resid_blocks = np.copy(resid_blocks)
        resid_blocks = compress.nbits_cost(resid_blocks)
        resid_blocks = np.minimum(resid_blocks, max_nbits)

        resids = resid_blocks.ravel()
        raw_resids = np.copy(resids)

        block_maxes = np.max(resid_blocks, axis=1)

        sb.distplot(resids, ax=ax, kde=False, bins=bins, hist_kws={
            'normed': True, 'range': [0, max_nbits]})

        sb.distplot(block_maxes, ax=ax, kde=False, bins=bins, hist_kws={
            'normed': True, 'range': [0, max_nbits], 'color': (.8, 0, 0, .05)})

        if plot_sort:
            positions = sort_transform(raw_resids, nbits=numbits, prefix_nbits=8).ravel()
            position_costs = compress.nbits_cost(positions, signed=False)
            sb.distplot(position_costs, ax=ax, kde=False, bins=bins, hist_kws={
                'normed': True, 'range': [0, max_nbits], 'color': 'g'})

        if plot_mixfix:
            # print "raw resids[2, :20]: ", raw_resid_blocks[:2]
            costs_mixfix = mixfix_cost(raw_resid_blocks, nbits=numbits).ravel()
            sb.distplot(costs_mixfix, ax=ax, kde=False, bins=bins, hist_kws={
                'normed': True, 'range': [0, max_nbits], 'color': 'g'})

        if plot_sub_minbits:
            use_blocks = zigzag_encode(raw_resid_blocks)
            sub_minbits_block_costs = compress.nbits_cost(use_blocks)
            # sub_minbits_min_costs = np.min(sub_minbits_block_costs, axis=1, keepdims=True)
            sub_minbits_max_costs = np.max(sub_minbits_block_costs, axis=1, keepdims=True)
            # costs_sub_minbits = (sub_minbits_block_costs - sub_minbits_min_costs).ravel()
            cost_diffs = sub_minbits_block_costs - sub_minbits_max_costs
            # largest value -> 0; smaller values -> -4; then add 4
            costs_sub_minbits = 4 + np.maximum(-4, cost_diffs).ravel()
            sb.distplot(costs_sub_minbits, ax=ax, kde=False, bins=bins, hist_kws={
                'normed': True, 'range': [0, max_nbits], 'color': 'gray'})

            # positions = sort_transform(raw_resids, nbits=numbits, prefix_nbits=8).ravel()
            # position_costs = compress.nbits_cost(positions, signed=False)

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
        if plot_mixfix:
            costs.append(np.mean(costs_mixfix))
            styles.append('g--')
        if plot_sub_minbits:
            costs.append(np.mean(costs_sub_minbits))
            styles.append('gray')
        zipped = zip(costs, styles)
        for i, (cost, style) in enumerate(zipped):
            # cost = costs[i]
            # style = styles[i]
            ax.plot([cost, cost], [0, 1], style, lw=1)
            x = cost / float(xlim[1])
            if len(zipped) > 3:
                y = .05 + (.8 * (i % 2)) + (.025 * (i % 4))  # stagger heights
            else:
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
    # plot_uresiduals_bits(errs_left, ax=axes[3])
    # plot_uresiduals_bits(errs_right, ax=axes[-1])
    plot_uresiduals_bits(offsetBlocksLeft, ax=axes[3])
    plot_uresiduals_bits(offsetBlocksRight, ax=axes[-1])

    _, ymax_left = axes[3].get_ylim()
    _, ymax_right = axes[-1].get_ylim()
    ymax = max(ymax_left, ymax_right)
    [ax.set_ylim([0, ymax]) for ax in (axes[3], axes[-1])]

    fig.tight_layout()


def main():
    np.set_printoptions(formatter={'float': lambda x: '{:.3f}'.format(x)})

    # # # uncomment this to get the LUT for mapping comparison bytes to indices
    # lut, perms = create_perm_lut()
    # print "perm lut, perms: ", lut, "\n", perms
    # assert len(np.unique(lut)) == 25
    # import sys
    # sys.exit()

    # dsets = ds.smallUCRDatasets()

    # dset = dslist[2]
    # X = dset.X

    # print "learning dict seq for dataset: ", dset.name
    # learn_dict_seq(X, max_len=1024)

    # import sys
    # sys.exit()

    # mpl.rcParams.update({'figure.autolayout': True})  # make twinx() not suck

    # ------------------------ experimental params

    # left_transforms = None
    # left_transforms = 'sub_mean'
    # left_transforms = 'dyn_filt'
    left_transforms = 'delta'
    # left_transforms = ['smooth', 'delta']
    # left_transforms = ['smooth', 'smooth', 'delta']
    # left_transforms = 'double_delta'
    # left_transforms = 'online_regress'
    # left_transforms = 'global_regress'
    # left_transforms = 'VAR'
    # left_transforms = ['delta', 'blocklen=4']
    # left_transforms = 'dyn_delta'
    # left_transforms = ['dyn_delta', 'blocklen=4']
    # left_transforms = ['blocklen=2', 'dyn_delta', 'blocklen=8']
    # left_transforms = ['blocklen=4', 'dyn_delta', 'blocklen=8']
    # left_transforms = 'double_delta'  # delta encode deltas
    # left_transforms = ['dyn_delta', 'kmeans']

    # right_transforms = None
    # right_transforms = 'delta'
    # right_transforms = 'nn'
    # right_transforms = 'nn7'
    # right_transforms = 'double_delta'  # delta encode deltas
    # right_transforms = '1.5_delta'  # sub half of prev delta from each delta
    # right_transforms = 'dyn_delta'  # pick single or double delta for each block
    # right_transforms = 'dyn_filt'
    # right_transforms = 'online_regress'
    # right_transforms = 'global_regress'
    right_transforms = 'OnlineGradDescent'
    # right_transforms = ['smooth', 'OnlineGradDescent']
    # right_transforms = ['smooth', 'smooth', 'OnlineGradDescent']
    # right_transforms = ['downsample2', 'blocklen=8', 'OnlineGradDescent']
    # right_transforms = 'online_linreg'
    # right_transforms = 'VAR'
    # right_transforms = 'hash'
    # right_transforms = ['delta', 'blocklen=32']
    # right_transforms = ['delta', 'blocklen=16']
    # right_transforms = ['VAR', 'blocklen=16']
    # right_transforms = ['delta', 'blocklen=4']
    # right_transforms = ['delta', 'autocoracle']
    # right_transforms = ['delta', 'blocklen=2', 'kmeans']
    # right_transforms = ['delta', 'blocklen=2', 'online_kmeans']
    # right_transforms = ['dyn_delta', 'blocklen=2', 'online_kmeans']
    # right_transforms = ['dyn_delta', 'prefix_lut']
    # right_transforms = ['dyn_delta', 'sort_transform']
    # right_transforms = ['prefix_lut', 'dyn_delta']
    # right_transforms = ['dyn_delta', 'blocklen=4']
    # right_transforms = ['dyn_delta', 'blocklen=4', 'online_kmeans']
    # right_transforms = ['blocklen=4', 'dyn_filt']
    # right_transforms = ['blocklen=1', 'dyn_delta']
    # right_transforms = ['blocklen=2', 'dyn_delta', 'blocklen=8']
    # right_transforms = ['blocklen=4', 'dyn_delta', 'blocklen=8']
    # right_transforms = ['blocklen=16', 'dyn_delta', 'blocklen=8']
    # right_transforms = ['blocklen=16', 'dyn_delta', 'blocklen=8']
    # right_transforms = ['blocklen=8', 'dyn_filt', 'blocklen=8']
    # right_transforms = ['autocoracle', 'delta']
    # right_transforms = ['nn', 'dyn_delta']
    # right_transforms = ['delta', 'nn']
    # right_transforms = ['delta', 'nn', 'maybe_delta']
    # right_transforms = ['delta', 'kmeans']
    # right_transforms = ['dyn_delta', 'kmeans']
    # right_transforms = ['delta', 'online_kmeans']
    # right_transforms = ['dyn_delta', 'online_kmeans']
    # right_transforms = ['dyn_delta', 'blocklen=4']
    # right_transforms = ['dyn_delta', 'blocklen=2']
    # right_transforms = ['dyn_delta', 'blocklen=16']
    # right_transforms = ['delta', 'nn']
    # right_transforms = ['nn']
    # right_transforms = ['delta', 'nn']
    # right_transforms = ['nn', 'delta']
    # right_transforms = ['nn', 'dyn_delta']
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
    # right_transforms = 'dyn_delta'
    # right_transforms = 'dyn_filt'
    # right_transforms = 'split_dyn_filt'
    # right_transforms = ['delta', 'dyn_filt']
    # right_transforms = ['delta', 'split_dyn_filt']

    # numbits = 60  # 63 and 64 break our quantization somehow
    # numbits = 24
    numbits = 16
    # numbits = 12
    # numbits = 8

    # num_neighbors = 256
    num_neighbors = 1024

    neighbor_scales = None
    # neighbor_scales = [(0, 1), (1, 0)]  # amount to left shift, right shift
    # neighbor_scales = [.5, .75, 1.25, 1.5, 1.75, 2]
    # neighbor_scales = [.5, 2]

    invert_neighbors = False
    # invert_neighbors = True

    # nn_loss = 'l2'
    nn_loss = 'linf'

    # mean_normalize = True
    mean_normalize = False

    nn_nblocks = 1
    # nn_nblocks = 2

    predict_next = False
    # predict_next = True

    # prefix_nbits = 8
    # prefix_nbits = 10
    # prefix_nbits = 11
    prefix_nbits = -1

    k_left = -1
    k_right = -1
    # k = 4
    # k = 8
    # k = 16
    k = 32
    # k = 64
    # k = 128
    # k = 256
    # k = 1024
    # k_left = 256
    # k_right = 32

    # n = 2
    n = 8
    # n = 32
    # n = 100
    # n = 200
    # n = 500

    chunk_sz = -1
    # chunk_sz = 8
    # chunk_sz = 16
    # chunk_sz = 64
    # chunk_sz = 256

    save = True
    # save = False
    small = False
    # small = True

    dsets = []

    # TODO add in a "transform" that encodes blocks using codes we would use
    #   -actually, prolly bake this into nbits_cost cuz otherwise leading 0s
    #   and more than one leading 1 will make it underestimate costs

    # from python.datasets import uci_gas
    # dsets = uci_gas.all_recordings()
    # base_save_dir = 'uci_gas'

    # from python.datasets import pamap
    # dsets = pamap.all_recordings()
    # base_save_dir = 'pamap'

    # from python.datasets import msrc
    # # dsets = msrc.all_recordings(idxs=np.arange(30))
    # dsets = msrc.all_recordings(idxs=np.arange(10, 30))
    # base_save_dir = 'msrc'

    # from python.datasets import ampds
    # base_save_dir = 'ampds'
    # # dsets = ampds.all_power_recordings()
    # # dsets = ampds.all_gas_recordings()
    # # dsets = ampds.all_water_recordings()
    # # dsets = ampds.all_weather_recordings()
    # dsets = ampds.all_timestamp_recordings(); numbits = 24  # noqa

    #
    # NOTE: have to uncomment this for multivariate datasets
    #
    for ds in dsets:
        subseq_len = 500
        flat_data = ds.data.T.ravel()
        keep_len = int(len(flat_data) / subseq_len) * subseq_len
        flat_data = flat_data[:keep_len]
        ds.X = flat_data.reshape(-1, subseq_len)
        ds.X -= np.mean(ds.X, axis=1, keepdims=True)  # mean norm for better plotting
        # ds.X = compress_bench.concat_and_interpolate(ds.X)
        print "{} raveled data shape: {}".format(ds.name, ds.X.shape)
        print np.max(ds.X)

    # dsets = ucr.smallUCRDatasets() if small else ucr.origUCRDatasets()
    dsets = ucr.smallUCRDatasets() if small else ucr.allUCRDatasets()
    base_save_dir = 'ucr'

    # ds = dsets[0]
    # print "first ds: ", ds.name
    # print "original data:"
    # print ds.X.ravel()[:50]
    # print "quantized data:"
    # print quantize(ds.X, numbits=16, keep_nrows=1)[:50]
    # return

    # return
    # data_mats = [r.data for r in uci_gas.all_recordings()]
    # data_mat = np.hstack(data_mats)
    # dsets =

    # ------------------------ name output dir based on params
    suffix = ""
    prefix = "small-" if small else ""
    if "nn" in right_transforms:
        suffix = "_nn{}".format(num_neighbors)
        suffix += "_inv{}".format(1 if invert_neighbors else 0)
        if neighbor_scales:
            legible_scales = ','.join([str(s) for s in neighbor_scales])
            suffix += "_scale{}".format(legible_scales)
        suffix += "_loss={}".format(nn_loss)
        suffix += "_meannorm" if mean_normalize else "_nonorm"
        suffix += '' if nn_nblocks < 2 else '_nnblocks={}'.format(nn_nblocks)
        suffix += '_forecast' if predict_next else ''
    if "kmeans" in right_transforms or "online_kmeans" in right_transforms:
        suffix += '_k={}'.format(k)
    if n < 32:
        suffix += "_n={}".format(n)
    if prefix_nbits > 0:
        suffix += '_prefix={}b'.format(prefix_nbits)
    if chunk_sz > 0:
        suffix += '_chunk={}'.format(chunk_sz)
    transforms_str = "{}-vs-{}".format(name_transforms(left_transforms),
                                       name_transforms(right_transforms))
    subdir = '{}{}b_{}{}'.format(prefix, numbits, transforms_str, suffix)
    subdir = os.path.join(base_save_dir, subdir)
    print "saving to subdir: {}".format(subdir)

    # ------------------------ main loop

    # if True:
    if False:
        from .utils import distance as dist
        from .utils import sliding_window as win
        # k = 16
        k = 128
        n = 16
        # n = 32
        block_sz = 8
        block_sz = 2

        subdir = 'centroids'
        if block_sz != 8:
            subdir += '_blocksz={}'.format(block_sz)
        subdir += '_k={}'.format(k)

        # for d in dslist[26:27]:  # olive oil
        # for d in dslist[:5]:
        # for d in dslist[0:1]:
        # for d in dslist:
        for d in dslist[7:15]:
            data = d.X
            # print "X.shape", X.shape
            data = quantize(data, numbits, keep_nrows=n)
            # mats = (data, delta_encode(data))
            blocks = convert_to_blocks(data)
            rhs_data = dyn_delta_encode(blocks)
            rhs_data = rhs_data.reshape(data.shape)
            mats = (data, rhs_data)
            names = ('raw', 'dyn deltas (zoomed)')
            fig, all_axes = plt.subplots(3, 2, figsize=(11, 8))

            print "---- ", d.name

            for i, X in enumerate(mats):
                name = names[i]
                axes = all_axes[:, i]

                # print "X.shape", X.shape
                all_windows = [win.sliding_window_1D(ts, block_sz) for ts in X]
                all_windows = np.vstack(all_windows)
                # print "X.shape", X.shape
                centroids, assigs = dist.kmeans(all_windows, k=k)

                # print "all_windows min, max", np.min(all_windows), np.max(all_windows)
                # print "centroids min, max", np.min(centroids), np.max(centroids)

                axes[0].set_title(d.name + ' ' + name)
                axes[1].set_title("{} kmeans centroids".format(k))
                axes[0].plot(X.T, lw=1)
                if block_sz > 2:
                    axes[1].plot(centroids.T)
                else:
                    # sb.jointplot(centroids[:, 0], centroids[:, 1], ax=axes[1])
                    x, y = centroids[:, 0], centroids[:, 1]
                    axes[1].scatter(x, y)

                    # fit with np.polyfit
                    m, b = np.polyfit(x, y, 1)
                    axes[1].plot(x, m*x + b, 'k--', lw=1)
                    r = np.corrcoef(x, y)[0, 1]  # take off diag of corr mat
                    axes[1].set_title(axes[1].get_title() + ' (r = {:.2f})'.format(r))

                low, high = np.percentile(X, [1, 99])
                axes[0].set_ylim((low, high))

                X_log = np.log2(np.abs(X)) * np.sign(X)
                # axes[2].plot(compress.nbits_cost(X[:8].T), lw=1)
                axes[2].plot(X_log.T, lw=1)

                plt.tight_layout()
            save_current_plot(d.name, subdir=subdir)
        # plt.show()
        return

    dslist = list(dsets)
    # for d in dsets:
    # for d in dslist[4:8]:
    # for d in dslist[26:29]:
    # for d in dslist[20:21]: # ItalyPowerDemand
    # for d in dslist[12:13]:  # ECGFiveDays
    # for d in dslist[:4]:
    # for d in dslist[4:5]:  # ChlorineConcentration
    # for d in dslist[23:24]:  # MALLAT
    # for d in dslist[:23]:
    # for d in dslist[6:23]:
    # for d in dslist[:5]:
    # for d in dsets:
    # for d in dslist[31:]:
    # for d in dslist[:1]: # 50words
    # for d in dslist[26:27]:  # OSULeaf
    # for d in dslist[27:28]:  # Olive Oil
    # for d in dslist[30:31]:  # starlight curves
    # for d in dslist[2:3]:  # beef
    # for d in dslist[4:6]:
    # for d in dslist[3:4]: # CBF
    # for d in (dslist[26], dslist[27], dslist[30], dslist[1], dslist[14]):
    # for d in dslist[:31]:
    # for d in dslist[1:2]:  # adiac
    for d in dslist:
        print "------------------------ {}".format(d.name)
        # continue # TODO rm
        plot_dset(d, numbits=numbits, n=n,
                  left_transforms=left_transforms,
                  right_transforms=right_transforms,
                  num_neighbors=num_neighbors,
                  scale_factors=neighbor_scales,
                  try_invert=invert_neighbors,
                  nn_loss=nn_loss,
                  nn_nblocks=nn_nblocks,
                  mean_normalize=mean_normalize,
                  predict_next=predict_next,
                  prefix_nbits=prefix_nbits,
                  k=k, k_left=k_left, k_right=k_right,
                  chunk_sz=chunk_sz)
        if save:
            save_current_plot(d.name, subdir=subdir)

    if not save:
        plt.show()


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    main()
