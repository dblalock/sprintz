#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import tempfile

from python import dfset
from python import codec


pd.options.mode.chained_assignment = None  # default='warn'


def encode(dfs, codeclist):
    """
    args:
        dfs (DfSet): wrapper around a collection of DataFrames; can
            efficiently read and write individual columns for individual
            dataframes. Modified inplace.
        codeclist (iterable of BaseCodec): sequence of transformations
            to apply to each DataFrame. Any codec with needs_train = True is
            trained inplace.
    returns:
        list of (dict: dfid -> header). There is one dict per codec in
            `codecs`. Each dict has one entry per DataFrame in `dfs`. Each
            header contains necessary info specific to each codec.
    """
    # train_dfs (DfSet, optional): separate set of DataFrames to train
    #         the encoders on. If None, encoders are trained on `dfs`. This is
    #         also modified inplace.
    # if train_dfs is None:
    #     train_dfs = dfs

    # TODO ideally look at dependency structure of which cols are read and
    # written to inform how dfs caches cols

    # all_headers = []
    dfids = dfs.ids

    # TODO do as much as possible before and after stuff that needs training
    # one df at a time (like in the `else`), rather than one enc at a time
    needs_train = any([est.needs_training() for est in codeclist])
    headers = {}
    if needs_train:
        for est in codeclist:
            if est.needs_training():
                traincols = est.train_cols()
                for dfid in dfids:
                    est.train(dfs[dfid, traincols])
            for dfid in dfids:
                df = dfs[dfid, est.cols()]
                dirty_df, header = est.encode(df)
                headers[dfid] = headers.get(dfid, []) + [header]
                # print("dirty_df.shape", dirty_df.shape)
                # print("dirty_df", dirty_df)
                dfs[dfid] = dirty_df
            # all_headers.append(headers)
    else:
        for dfid in dfids:
            headerlist = []
            for est in codeclist:
                # print("est cols: ", est.cols())
                df = dfs[dfid, est.cols()]
                dirty_df, header = est.encode(df)
                headerlist.append(header)
                dfs[dfid] = dirty_df
            headers[dfid] = headerlist

    return headers


def decode(dfs, codeclist, headers):
    """
    args:
        dfs (DfSet): wrapper around a collection of DataFrames; can
            efficiently read and write individual columns for individual
            dataframes. Modified inplace.
        codeclist (iterable of BaseCodec): sequence of transformations
            to apply to each DataFrame. These codec objects must be the same
            ones used in the corresponding call to `encode`.
        headerlist (iterable of dict: dfid -> params): the sequence of header
            dictionaries returned by the corresponding call to `encode`.
    returns:
        None
    """
    codeclist = codeclist[::-1]
    # headerlist = headerlist[::-1]

    ids = dfs.ids
    for dfid in ids:
        df = dfs[dfid]
        modified_cols = set()
        headerlist = headers[dfid][::-1]
        for i, est in enumerate(codeclist):
            header = headerlist[i]
            cols = est.cols() or df.columns
            dirty_df = est.decode(df[cols], header)  # writes inplace
            dirty_cols = dirty_df.columns
            for col in dirty_cols:
                df[col] = dirty_df[col]
            modified_cols |= set(dirty_cols)
        # print("dfid: ", dfid)
        # print("modified_cols: ", modified_cols)
        # print("should be about to write stuff...")
        # print("col a: ", df['a'])
        dfs[dfid] = df[list(modified_cols)]


# def encode_measure_decode(csvdir, dfsdir, codeclist, filetype='h5', **dfset_kwargs):
# def encode_measure_decode(dfs, codeclist, check_correct=True, **dfset_kwargs):
def encode_measure_decode(dfs, codeclist, check_correct=True):
    # dfs = dfset.make_dfset(csvdir, dfsdir, filetype=filetype, **dfset_kwargs)

    with tempfile.TemporaryDirectory() as dirpath:

        # print("dirpath: ", dirpath)

        # dfs_orig = dfset.make_dfset(
        #     dfsdir=dirpath, filetype=dfs.filetype)
        dfs_orig = dfs.copy(dfsdir=dirpath)
        if check_correct:
            dfs.clone_into(dfs_orig)
        # dfs_orig = dfset.make_dfset(csvdir, dfsdir.rstrip('/') + '_orig',
        #                             filetype=filetype, **dfset_kwargs)

        # print("about to measure file sizes")
        sizes_df_orig = dfs.file_sizes()
        # print("about to encode")
        headerlist = encode(dfs, codeclist)
        # print("about to re-measure file sizes")
        sizes_df_comp = dfs.file_sizes()

        if check_correct:

            # print("a orig: ", dfs_orig['df0', 'a'])
            # print("a comp: ", dfs['df0', 'a'])
            decode(dfs, codeclist, headerlist)
            # print("a hat:  ", dfs['df0', 'a'])
            sizes_df_decomp = dfs.file_sizes()

            sizes_df_orig.sort_values(['dfid', 'col'], axis=0, inplace=True)
            sizes_df_decomp.sort_values(['dfid', 'col'], axis=0, inplace=True)
            sizes_orig = sizes_df_orig['nbytes'].values
            sizes_decomp = sizes_df_decomp['nbytes'].values

            # print('sizes_orig', sizes_orig)
            # print('sizes_comp', sizes_comp)
            # print('sizes_decomp', sizes_decomp)
            # first sanity check file sizes
            assert np.array_equal(sizes_orig, sizes_decomp)
            # full equality comparison
            dfs_orig.equals(dfs, raise_error=True)

    return sizes_df_orig, sizes_df_comp


def main():
    pass


if __name__ == '__main__':
    main()
