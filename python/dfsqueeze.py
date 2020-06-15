#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import tempfile

from python import dfset
from python import codec


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

    ret = []
    dfids = dfs.all_ids()
    for est in codeclist:
        headers = {}
        if est.needs_training:
            traincols = est.train_cols()
            for dfid in dfids:
                est.train(dfs[dfid, traincols])
        for dfid in dfids:
            df, header = est.encode(dfs[dfid, est.cols()])
            headers[dfid] = header
            dfs[dfid, est.write_cols()] = df
        ret.append(headers)

    return ret


def decode(dfs, codeclist, headers):
    """
    args:
        dfs (DfSet): wrapper around a collection of DataFrames; can
            efficiently read and write individual columns for individual
            dataframes. Modified inplace.
        codeclist (iterable of BaseCodec): sequence of transformations
            to apply to each DataFrame. These codec objects must be the same
            ones used in the corresponding call to `encode`.
        headers (iterable of dict: dfid -> params): the sequence of header
            dictionaries returned by the corresponding call to `encode`.
    returns:
        None
    """
    ids = dfs.all_ids()
    for dfid in ids:
        df = dfs[dfid]
        modified_cols = set()
        for i, est in enumerate(codeclist):
            header = headers[i]
            write_cols = est.write_cols()
            est.decode(df[est.cols()], header)  # modifies df inplace
            # updated_df = est.decode(df[est.cols()], header)
            # # TODO can we rely on decode to just modify these series inplace?
            # if write_cols is None:
            #     df = updated_df
            # else:
            #     df[write_cols] = updated_df

            # update set of modified cols (so we don't necessarily have to
            # write back everything)
            if modified_cols is None or write_cols is None:
                modified_cols = None  # None = all cols
            else:
                modified_cols |= set(write_cols)
        write_df = df if modified_cols is None else df[list(modified_cols)]
        dfs[dfid, modified_cols] = write_df


# def encode_measure_decode(csvdir, dfsdir, codeclist, filetype='h5', **dfset_kwargs):
def encode_measure_decode(dfs, codeclist, filetype='h5', check_correct=True,
                          **dfset_kwargs):
    # dfs = dfset.make_dfset(csvdir, dfsdir, filetype=filetype, **dfset_kwargs)

    with tempfile.TemporaryDirectory() as dirpath:
        dfs_orig = dfset.make_dfset(dfsdir=dirpath,
                                    filetype=filetype, **dfset_kwargs)
        if check_correct:
            dfs.clone_into(dfs_orig)
        # dfs_orig = dfset.make_dfset(csvdir, dfsdir.rstrip('/') + '_orig',
        #                             filetype=filetype, **dfset_kwargs)

        sizes_df_orig = dfs.file_sizes()
        encode(dfs, codeclist)
        sizes_df_comp = dfs.file_sizes()

        if check_correct:
            decode(dfs, codeclist)
            sizes_df_decomp = dfs.file_sizes()

            sizes_df_orig.sort_values(['dfid', 'col'], axis=0, inplace=True)
            sizes_df_decomp.sort_values(['dfid', 'col'], axis=0, inplace=True)
            sizes_orig = sizes_df_orig['nbytes'].values
            sizes_decomp = sizes_df_decomp['nbytes'].values

            # first sanity check file sizes
            assert np.array_equal(sizes_orig, sizes_decomp)
            # full equality comparison
            dfs_orig.equals(dfs, raise_error=True)

    return sizes_df_orig, sizes_df_comp


def main():
    pass


if __name__ == '__main__':
    main()
