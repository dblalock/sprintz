#!/usr/bin/env python

import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

from ..utils.files import ensure_dir_exists

# funcs for plotting datasets that are toot tightly coupled to assumptions
# about how we store the data to be in the utils directory


def _plot_corr(data, fig, ax):
    """assumes data is row-major; ie, each col is one variable over time"""
    # cov = np.cov(data.T)
    corr = np.corrcoef(data.T)
    # im = ax.imshow(corr, interpolation='nearest',
    #                cmap=plt.cm.RdBu,
    #                norm=mpl.colors.Normalize(vmin=-1., vmax=1.))
    # fig.colorbar(im, ax=ax)
    # sb.heatmap(corr, center=0, ax=ax, square=True)
    sb.heatmap(corr, vmin=-1, vmax=1, center=0, ax=ax, square=True)


def plot_recordings(recordings, interval_len=1000, norm_means=False,
                    mins_zero=False, savedir=None):

    for r in recordings:
        print "recording {} has data of shape {}".format(r.name, r.data.shape)
        fig, axes = plt.subplots(2, 4, figsize=(13, 7))

        start_idxs = [0, len(r.data) - interval_len]
        end_idxs = [interval_len, len(r.data)]

        # any_nans_in_row = np.isnan(r.data).sum(axis=1)
        # print np.where(any_nans_in_row)[0]
        # continue

        cor_sample_length = len(r.data) / 5  # semi-arbitrary, for corr mat

        for i, (start, end) in enumerate(zip(start_idxs, end_idxs)):
            timestamps = r.sampleTimes[start:end]
            data = r.data[start:end]
            if norm_means:
                data -= np.mean(data, axis=0).astype(data.dtype)
            elif mins_zero:
                data -= np.min(data, axis=0).astype(data.dtype)

            # print "data shape", data.shape
            # print "data final vals", data[-20:]
            # continue

            col = i + 1
            axes[0, col].plot(timestamps, data, lw=1)
            axes[1, col].plot(timestamps[1:], np.diff(data, axis=0), lw=1)
            axes[0, col].set_title('data')
            axes[1, col].set_title('first derivs')

        # plot correlation matrices for orig data and first derivs
        cor_sample_length = len(r.data) / 5
        data = r.data[:cor_sample_length]
        _plot_corr(data, fig, axes[0, 0])
        _plot_corr(np.diff(data, axis=0), fig, axes[1, 0])
        data = r.data[-cor_sample_length:]
        _plot_corr(data, fig, axes[0, -1])
        _plot_corr(np.diff(data, axis=0), fig, axes[1, -1])

        # _plot_corr(r.data[:cor_sample_length], fig, axes[0, 0])
        # data = r.data[-cor_sample_length:]
        # _plot_corr(data, fig, axes[2, 1])

        plt.tight_layout()
        # plt.show()

        if savedir is not None:
            ensure_dir_exists(savedir)
            plt.savefig(os.path.join(savedir, r.name))
