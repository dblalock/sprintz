#!/usr/bin/env python

from __future__ import absolute_import

import os
import numpy as np
import pandas as pd
import unittest

from python import dfset
from python import dfsqueeze as sq
from python import codec


MOCK_IN_DIR = 'debug_in'
MOCK_OUT_DIR = 'debug_out'
MOCK_CONTEXT_PATH = 'debug_context.csv'

# TODO create and rm these in setup and teardown
for subdir in [MOCK_IN_DIR, MOCK_OUT_DIR]:
    if not os.path.exists(subdir):
        os.makedirs(subdir)


def _debug_df0():
    a = np.arange(4) + 10
    b = a[::-1].copy()
    c = np.array([-1, 1, -2, 2])
    return pd.DataFrame.from_dict(dict(a=a, b=b, c=c))


def _debug_df1():
    a = np.arange(4) - 3.5
    b = np.array([100.12, -100.34, 100.56, -100.78])
    d = np.array([5, -4, 3, -2])
    return pd.DataFrame.from_dict(dict(a=a, b=b, d=d))


def _populate_mock_input_dir(df0=None, df1=None):
    if df0 is None:
        df0 = _debug_df0()
    if df1 is None:
        df1 = _debug_df1()
    df0.to_csv(os.path.join(MOCK_IN_DIR, 'df0.csv'), index=False)
    df1.to_csv(os.path.join(MOCK_IN_DIR, 'df1.csv'), index=False)


def _rm_mock_input_files():
    os.remove(os.path.join(MOCK_IN_DIR, 'df0.csv'))
    os.remove(os.path.join(MOCK_IN_DIR, 'df1.csv'))


def main():
    _populate_mock_input_dir()

    pipelines = []
    # pipelines.append([codec.Delta()])
    # pipelines.append([codec.Delta(), codec.Zigzag()])

    # just quantize
    pipelines.append([])
    # quantize and bz2
    pipelines.append([codec.Bzip2()])
    # quantize, byteshuf, bz2
    pipelines.append([codec.ByteShuffle(), codec.Bzip2()])
    # quantize + {dynamic,double,plain}-delta code, with and without byteshuf
    pipelines.append([codec.Delta(), codec.Zigzag(), codec.Bzip2()])  # noqa
    pipelines.append([codec.DoubleDelta(), codec.Zigzag(), codec.Bzip2()])  # noqa
    pipelines.append([codec.DynamicDelta(), codec.Zigzag(), codec.Bzip2()])  # noqa
    pipelines.append([codec.Delta(), codec.Zigzag(), codec.ByteShuffle(), codec.Bzip2()])  # noqa
    pipelines.append([codec.DoubleDelta(), codec.Zigzag(), codec.ByteShuffle(), codec.Bzip2()])  # noqa
    pipelines.append([codec.DynamicDelta(), codec.Zigzag(), codec.ByteShuffle(), codec.Bzip2()])  # noqa

    csearch = codec.CodecSearch(pipelines=pipelines, loss='nbytes')

    codeclist = [codec.Quantize(), csearch]

    dfs = dfset.make_dfset(filetype='parquet', csvsdir=MOCK_IN_DIR)
    sizes_df_orig, sizes_df_comp = sq.encode_measure_decode(
            dfs, codeclist)


    # TODO construct a bunch of pipelines of preprocs and formats/compressors,
    # get the sizes for every (dfid, col) combo, dump to a file, and then
    # make simple plots or something breaking down results
    #   -tricky part is that we might wanna try different combos of stuff
    #   on different cols
    #       -in particular, colsum codec is weird
    #   -need a class to try out different pipelines on each col and use the
    #   best
    #       -actually, do we need this? if just want plots or something, can
    #       we just figure out what best would have been?
    #           -no, this just tells you conditional averages of each var;
    #           want to know actual overall compression of each df when you
    #           try "for real" to get as much compression as possible
    #       -would be nice to have it try running a real compressor on the
    #       results as its loss, instead of just some proxy

    # SELF: pick up by adding this class to codec.py
        # then construct some pipelines and run them
        # also, have dyndelta actually bitpack its masks


    _rm_mock_input_files()



if __name__ == '__main__':
    main()
