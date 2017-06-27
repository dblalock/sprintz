#!/usr/bin/env python

from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np


def _imshow_better(X, ax=None):
    if not ax:
        ax = plt
    ax.imshow(X, interpolation='nearest', aspect='auto')


_Edge = namedtuple('Edge', 'ts', 'te', 'cost')


def pla_anchors(signal, max_err):  # TODO not just L2 error
    # assert len(signal) % 2 == 0  # TODO rm even length constraint

    N = len(signal)
    edges = [_Edge(i, i+1, 0.) for i in range(N - 1)]

    qu = []

    # TODO finish this


# from flock paper
def optimalAlignment(scores, m, scoresForEndIdxs=True):
    """
    Given an array of scores for the end positions of subsequences of length m
    (where higher scores correspond to better alignments and negative scores
    indicate locations that should be "skipped"), returns the end indices of
    the optimal placement of subsequences.

    Parameters
    ------
    scores: a vector of scores
    m: length of the previous values with which overlap is disallowed

    >>> s = [2,1,3,4]
    >>> optimalAlignment(s, 2)
    array([0, 3])
    >>> s = [1,2,3,2,5,1]
    >>> optimalAlignment(s, 2)
    array([0, 2, 4])
    >>> s = [-1,2,3,2,5,1]
    >>> optimalAlignment(s, 2)
    array([2, 4])
    >>> optimalAlignment(s, 2, scoresForEndIdxs=False)
    array([2, 4])
    """
    if scores is None or not len(scores):
        return None

    scores = np.asarray(scores)
    n = len(scores)

    if n <= m:
        return np.argmax(scores)

    # if the scores are associate with start indices, rather than end
    # indices, we need to ensure things don't overlap with positions after
    # them, not positions before them; the easiest way to do this is just
    # to reverse the order of everything
    #
    # TODO almost positive we can remove this param--algo guarantees that
    # scores taken have gaps of at least m-1 between them; doesn't care
    # about not overlapping before vs after
    if not scoresForEndIdxs:
        scores = scores[::-1]

    c = np.empty(scores.shape)                          # cumulative score
    c[:m] = scores[:m]
    parentIdxs = np.empty(scores.shape, dtype=np.int)   # previous best idx
    parentIdxs[:m] = -1

    idx_bsf = -1    # best-so-far
    score_bsf = 0.

    # compute best parent at least m time steps ago for every idx;
    # a "parent" is a previous cumulative score we can add to
    for i in range(m, n):
        oldIdx = i - m
        if c[oldIdx] > score_bsf:
            idx_bsf = oldIdx
            score_bsf = c[oldIdx]
        c[i] = scores[i] + score_bsf
        parentIdxs[i] = idx_bsf

    # compute lineage of best score
    parent = np.argmax(c)
    if scores[parent] < 0.:  # edge case: all scores are negative
        return None
    parents = []
    while parent >= 0:
        parents.append(parent)
        parent = parentIdxs[parent]

    parents = np.array(parents, dtype=np.int)
    if scoresForEndIdxs:
        return parents[::-1]  # appended in reverse order above
    else:
        return (n-1) - parents  # we reversed the order at the start


def sq_loss(signal, i, j, penalty=-1):
    # uses signal[i:(j+1)]

    # TODO precompute cumsums, cumsums sq and create loss func via
    # factory func

    start = signal[i]
    end = signal[j]
    slope = (end - start) / (j - i)

    # interp = np.arange(start + slope, end - .0001, slope)
    interp = (np.arange(1, j - i) * slope) + start
    diffs = signal[(i + 1):j] - interp

    # PENALTY = np.std(signal)  # fixed penalty for adding a line
    # PENALTY = np.std(signal) / 2  # fixed penalty for adding a line
    if penalty < 0:
        penalty = np.std(signal)

    return np.sum(diffs * diffs) + penalty


def optimal_pla(signal, loss_func=sq_loss, **loss_func_kwargs):
    if signal is None or len(signal) == 0:
        return None
    signal = np.asarray(signal)
    assert len(signal.shape) == 1  # must be 1D

    N = len(signal)

    parents = np.zeros(N, dtype=np.int32)
    cumloss = np.full(N, np.inf, dtype=np.float32)

    parents[0] = -1
    cumloss[0] = 0
    parents[1] = 0
    cumloss[1] = loss_func(signal, 0, 1)

    # TODO rm
    raw_cumloss = np.zeros((N, N), dtype=np.float32)
    raw_cumloss[:, 0] = 0
    raw_cumloss[0, 1] = loss_func_kwargs['penalty']

    # compute best parent and loss for each idx
    for n in range(2, N):
        for m in range(n):
            loss = loss_func(signal, m, n, **loss_func_kwargs) + cumloss[m]
            if loss < cumloss[n]:
                cumloss[n] = loss
                parents[n] = m

            # TODO rm
            raw_cumloss[m, n] = loss

    # trace the best path back from the final index
    anchor = N-1
    anchors = []
    while anchor >= 0:
        anchors.append(anchor)
        anchor = parents[anchor]

    # # compute how good previous solutions were
    # _, axes = plt.subplots(2, figsize=(6, 8))
    # min_prev_losses = np.copy(raw_cumloss)
    # for j in range(2, n):
    #     col = raw_cumloss[:j, j]
    #     minidx = np.argmin(col)
    #     min_prev_losses[minidx:j, j] = col[minidx]

    # _imshow_better(np.log2(raw_cumloss), ax=axes[0])
    # _imshow_better(np.log2(min_prev_losses), ax=axes[1])
    # # plt.colorbar()

    return np.array(anchors, dtype=np.int32)[::-1], cumloss[N-1]


# # see SlidingStraightLineProjection in ts representation.py
# # EDIT: don't actually need this func unless we want to tighten
# # bound with SSE; only comes up in 2/8 cases...
# def sliding_line_sse(y, windowlen):
#     m = windowlen

#     # x = np.arange(m)
#     # https://proofwiki.org/wiki/Sum_of_Sequence_of_Squares
#     x = np.arange(1, windowlen + 1)
#     sumX =
#     # sumX = np.sum(x)  # TODO compute directly from windowlen
#     # Sxx = np.sum(x*x) - sumX * sumX / windowlen

#     # cumX2 = np.cumsum(x*x)


#     cumY = np.cumsum(y)
#     cumY2 = np.cumsum(y*y)
#     # cumXY = np.cumsum(x*y)

#     n = len(y)
#     sse = np.empty(n - m + 1)

def main():

    _, axes = plt.subplots(2)

    x = np.random.randn(100)

    # penalty = np.var(x)
    # penalty = np.var(x) / 2
    penalty = np.var(x) / 4
    anchors0, _ = optimal_pla(x, penalty=penalty)
    axes[0].plot(np.copy(x))

    x = np.cumsum(x)
    x -= np.min(x)
    x = np.floor(x * (255 / np.max(x)))
    penalty = np.var(x) / 4
    anchors, loss = optimal_pla(x, penalty=penalty)
    axes[1].plot(x)

    print "found anchors: ", anchors
    print "got normalized loss {} using {} anchors".format(
        loss / (len(x) * np.var(x)), len(anchors))

    # plt.figure()
    # axes[0].plot(x)
    ylim = np.min(x), np.max(x)
    for i in range(len(anchors) - 1):
        xvals = anchors[i:(i + 2)]
        yvals = x[xvals]
        axes[1].plot(xvals, yvals, color='r')

        # for debugging
        axes[1].plot((xvals[0], xvals[0]), ylim, 'k--')

    # for ax in axes:
        # ax.set_ylim(ylim)
    plt.ylim(ylim)
    plt.show()


if __name__ == '__main__':
    main()
