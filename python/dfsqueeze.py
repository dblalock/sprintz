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
                dfs[dfid, dirty_df.columns] = dirty_df
            # all_headers.append(headers)
    else:
        for dfid in dfids:
            headerlist = []
            # df = dfs[dfid]
            # print("dfid: ", dfid)
            # import sys; sys.exit()
            for est in codeclist:
                # print("encode: est class, cols: ", type(est), est.cols())

                # print("col vals: ")
                # print("gps_speed_unreliable: ")
                # s = dfs[dfid, 'gps_speed_unreliable']
                # print(s, type(s), s.dtype)

                subdf = dfs[dfid, est.cols()]
                # print("about to encode df with cols: ", df.columns)
                dirty_df, header = est.encode(subdf)
                headerlist.append(header)
                # for col in dirty_df.columns:
                #     df[col] = dirty_df[col]

                print("encode: dtype in df right before writing: ", dirty_df['gps_lon'].dtype)

                dfs[dfid, dirty_df.columns] = dirty_df

                print("encode: dtype retrieved right after writing: ", dfs[dfid, 'gps_lon'].dtype)

            headers[dfid] = headerlist
            # dfs[dfid] = df

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
            # cols = est.cols() or df.columns
            # cols = est.cols()
            # if cols is not None:
            #     cols = sorted(list(set(cols) & set(df.columns)))
            # cols = est.cols_to_use(df)
            # dirty_df = est.decode(df[cols], header)  # writes inplace
            dirty_df = est.decode(df, header)  # writes inplace
            dirty_cols = dirty_df.columns
            for col in dirty_cols:
                df[col] = dirty_df[col]
            modified_cols |= set(dirty_cols)
        # print("dfid: ", dfid)
        # print("modified_cols: ", modified_cols)
        # print("should be about to write stuff...")
        # print("col a: ", df['a'])
        modified_cols = list(modified_cols)
        dfs[dfid, modified_cols] = df[modified_cols]


# def encode_measure_decode(csvdir, dfsdir, codeclist, filetype='h5', **dfset_kwargs):
# def encode_measure_decode(dfs, codeclist, check_correct=True, **dfset_kwargs):
def encode_measure_decode(dfs, codeclist, check_correct=True,
                          check_correct_inplace=False, check_file_sizes=False):
    # dfs = dfset.make_dfset(csvdir, dfsdir, filetype=filetype, **dfset_kwargs)

    with tempfile.TemporaryDirectory() as dirpath:

        # print("dirpath: ", dirpath)

        # dfs_orig = dfset.make_dfset(
        #     dfsdir=dirpath, filetype=dfs.filetype)
        if check_correct:
            dfs_orig = dfs.copy(dfsdir=dirpath)
            # dfs.clone_into(dfs_orig)
        # dfs_orig = dfset.make_dfset(csvdir, dfsdir.rstrip('/') + '_orig',
        #                             filetype=filetype, **dfset_kwargs)

        # print("about to measure file sizes")
        sizes_df_orig = dfs.file_sizes()
        # print("about to encode")
        print("================================ encode")
        headerlist = encode(dfs, codeclist)
        # print("about to re-measure file sizes")
        sizes_df_comp = dfs.file_sizes()

        if check_correct:

            with tempfile.TemporaryDirectory() as checkdir:
                if check_correct_inplace:
                    dfs_hat = dfs
                else:
                    dfs_hat = dfs.copy(dfsdir=checkdir)
                    print("copying dfs to dfs_hat")

                # print("a orig:\n", dfs_orig['df0', 'a'])
                # print("c orig:\n", dfs_orig['df0', 'c'])
                # print(dfs_orig['df0', 'c'].dtypes)
                # print("a comp: ", dfs['df0', 'a'])
                # print(dfs['df0', 'a'].dtypes)
                # print(dfs['df0', 'a'].dtypes)
                # print("c comp:\n", dfs['df0', 'c'])
                # print(dfs['df0', 'c'].dtypes)
                s = dfs['2137141111', 'gps_lon']
                print("gps_lon compressed dtype: ", s.dtype)

                print("================================ decode")
                decode(dfs_hat, codeclist, headerlist)
                print("================================ evaluate")
                # print("a hat:\n", dfs['df0', 'a'])
                # print(dfs['df0', 'a'].dtypes)
                # print("c hat:\n", dfs['df0', 'c'])
                # print(dfs['df0', 'c'].dtypes)
                # print("------------------------")
                sizes_df_decomp = dfs.file_sizes()

                # dfid = 'df1'
                # a_orig = dfs_orig[dfid, 'a']
                # a_hat = dfs[dfid, 'a']
                # assert np.allclose(a_orig, a_hat)

                sizes_df_orig.sort_values(
                    ['dfid', 'col'], axis=0, inplace=True)
                sizes_df_decomp.sort_values(
                    ['dfid', 'col'], axis=0, inplace=True)
                sizes_orig = sizes_df_orig['nbytes'].values
                sizes_decomp = sizes_df_decomp['nbytes'].values

                # print("sizes dfs:")
                # print(sizes_df_orig)
                # print(sizes_df_decomp)
                # print('sizes_orig', sizes_orig)
                # # print('sizes_comp', sizes_comp)
                # print('sizes_decomp', sizes_decomp)

                # full equality comparison
                dfs_orig.equals(dfs_hat, raise_error=True)

                # check file sizes
                # print("sizes orig, decomp:")
                # print(sizes_orig)
                # print(sizes_decomp)
                # vals0 = dfs['2137141111', 'lon']
                # vals1 = dfs_hat['2137141111', 'lon']
                # print(vals0.shape, vals0.values.itemsize)
                # print(vals1.shape, vals1.values.itemsize)
                if check_file_sizes:
                    # this only makes sense if the data for the dfs passed in
                    # was created with the same compression-related params as
                    # the dfs itself; this might be true, eg, if it got its
                    # data from a parquet dfset with no compression, while
                    # it was initialized to used compression; it will
                    # succesfully read those files, but their sizes won't match
                    # what it would have written itself (and by extension, what
                    # our dfs_hat writes)
                    assert np.array_equal(sizes_orig, sizes_decomp)

            #     print("does checkdir still exist?", os.path.exists(checkdir))
            # print("does dirpath still exist?", os.path.exists(dirpath))

    return sizes_df_orig, sizes_df_comp


def main():
    pass


if __name__ == '__main__':
    main()
