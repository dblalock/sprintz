#!/usr/bin/env python

from __future__ import absolute_import

import os
import numpy as np
import pandas as pd
import unittest

from python import dfset
from python import dfsqueeze as sq
from python import codec
from python import utils


MOCK_IN_DIR = 'debug_in'
MOCK_OUT_DIR = 'debug_out'
MOCK_CONTEXT_PATH = 'debug_context.csv'

# TODO create and rm these in setup and teardown
for subdir in [MOCK_IN_DIR, MOCK_OUT_DIR]:
    if not os.path.exists(subdir):
        os.makedirs(subdir)


def _debug_df0():
    a = np.arange(4) + 10
    assert a.dtype == np.int
    b = a[::-1].copy()
    c = np.array([-1, 1, -2, 2])
    return pd.DataFrame.from_dict(dict(a=a, b=b, c=c))


def _debug_df1():
    a = np.arange(4) - 3
    assert a.dtype == np.int
    b = np.array([100.12, -100.34, 100.56, -100.78])
    d = np.array([5, -4, 3, -2])
    return pd.DataFrame.from_dict(dict(a=a, b=b, d=d))


def _debug_df2():
    # float32 not preserved when written to csv; and since we write to csv
    # at the start of the associated tests to initialize the dfs, using float32
    # just makes tests fail cuz f32 gets turned to f64
    a = np.arange(4, dtype=np.float64)
    a[2] = np.nan
    b = a[::-1].copy()
    c = np.array([-1, 1, -2, 2])
    return pd.DataFrame.from_dict({'foo/a': a, 'bar/b/b': b, '/c': c})


def _populate_mock_input_dir(df0, df1):
    df0.to_csv(os.path.join(MOCK_IN_DIR, 'df0.csv'), index=False)
    df1.to_csv(os.path.join(MOCK_IN_DIR, 'df1.csv'), index=False)


def _rm_mock_input_files():
    for basename in os.listdir(MOCK_IN_DIR):
        path = os.path.join(MOCK_IN_DIR, basename)
        os.remove(path)
    # os.remove(os.path.join(MOCK_IN_DIR, 'df0.csv'))
    # os.remove(os.path.join(MOCK_IN_DIR, 'df1.csv'))


class DfsetTest(unittest.TestCase):

    def setUp(self):
        self.df0 = _debug_df0()
        self.df1 = _debug_df1()
        _populate_mock_input_dir(self.df0, self.df1)

    def tearDown(self):
        _rm_mock_input_files()


class TestDfSet(DfsetTest):

    def _test_dfs(self, filetype):
        dfs = dfset.make_dfset(filetype=filetype, csvsdir=MOCK_IN_DIR)
        # dfs = dfset.make_dfset(filetype=filetype)
        # print("dfs._dfsdir", dfs._dfsdir)
        # print("os.listdir(self._dfsdir)", os.listdir(dfs._dfsdir))
        # print("endswith:", dfs._endswith)
        # dfs.copy_from_csvs_dir(MOCK_IN_DIR)
        # print("os.listdir(self._dfsdir)", os.listdir(dfs._dfsdir))
        # print("dfs._dfsdir", dfs._dfsdir)
        # print("dfs.ids:", dfs.ids)
        # print("dfs._find_ids():", dfs._find_ids())

        assert sorted(dfs.ids) == ['df0', 'df1']
        # print(dfs._cols_stored_for_dfid('df0'))
        assert sorted(dfs._cols_stored_for_dfid('df0')) == 'a b c'.split()
        assert sorted(dfs._cols_stored_for_dfid('df1')) == 'a b d'.split()

        df0_hat = dfs['df0']
        df1_hat = dfs['df1']
        assert self.df0.shape == df0_hat.shape
        assert self.df1.shape == df1_hat.shape
        if filetype != 'csv':
            # print(self.df0.dtypes)
            # print(df0_hat.dtypes)
            # print(self.df0.index)
            # print(df0_hat.index)
            assert set(self.df0.dtypes) == set(df0_hat.dtypes)
            assert set(self.df1.dtypes) == set(df1_hat.dtypes)
        # print(self.df0.dtypes)
        # print(df0_hat.dtypes)
        assert set(self.df0.columns) == set(df0_hat.columns)
        assert set(self.df1.columns) == set(df1_hat.columns)
        for col in self.df0:
            assert np.array_equal(self.df0[col], df0_hat[col])
            assert np.allclose(self.df0[col], df0_hat[col], equal_nan=True)
        for col in self.df1:
            assert np.array_equal(self.df1[col], df1_hat[col])
            assert np.allclose(self.df1[col], df1_hat[col], equal_nan=True)

    def test_csv_dfs(self):
        self._test_dfs(filetype='csv')

    def test_npy_dfs(self):
        self._test_dfs(filetype='npy')

    def test_parquet_dfs(self):
        self._test_dfs(filetype='parquet')

    def test_feather_dfs(self):
        self._test_dfs(filetype='feather')

    def test_h5_dfs(self):
        self._test_dfs(filetype='h5')

    def test_santize_cols(self):
        _rm_mock_input_files()  # just want df2 here
        df = _debug_df2()
        dfpath = os.path.join(MOCK_IN_DIR, 'df2.csv')
        df.to_csv(dfpath, index=False)

        dfs = dfset.make_dfset(filetype='parquet', csvsdir=MOCK_IN_DIR)

        assert sorted(dfs.ids) == ['df2']
        assert set(dfs._cols_stored_for_dfid('df2')) == set(df.columns)
        df_hat = dfs['df2']
        assert set(df.columns) == set(df_hat.columns)
        assert df.shape == df_hat.shape
        assert set(df.dtypes) == set(df_hat.dtypes)

        for col in df:
            assert np.allclose(df[col], df_hat[col], equal_nan=True)

        os.remove(dfpath)

    def test_copy_from_csv_once(self):
        """We'd rather just convert csv to parquet or whatever one time;
        this test makes sure that that works"""
        dfsdir = 'dfs_test_copy_once'
        dfs_orig = dfset.make_dfset(dfsdir=dfsdir, filetype='parquet',
                                    csvsdir=MOCK_IN_DIR)
        dfs2 = dfset.make_dfset(dfsdir=dfsdir, filetype='parquet')
        assert dfs2.equals(dfs_orig)

        del dfs_orig
        dfs3 = dfset.make_dfset(dfsdir=dfsdir, filetype='parquet')
        assert dfs2.equals(dfs3)
        del dfs2
        dfs4 = dfset.make_dfset(dfsdir=dfsdir, filetype='parquet')
        assert dfs3.equals(dfs4)


class TestEncodeDecode(DfsetTest):

    def _test_filetype(self, filetype):
        dfs = dfset.make_dfset(filetype=filetype, csvsdir=MOCK_IN_DIR)

        encs = [codec.Debug(), codec.Delta()]
        # encs = [codec.Delta()]
        # encs = [codec.Debug()]

        sizes_df_orig, sizes_df_comp = sq.encode_measure_decode(
            dfs, encs)

    def test_all_filetypes(self):
        # for ftype in ('csv', 'npy'):
        # for ftype in ('csv',):
        for ftype in ('csv', 'npy', 'parquet', 'h5'):
            self._test_filetype(ftype)


class TestCodecs(DfsetTest):

    def _test_codecs_for_filetype(self, filetype, codeclist):
        dfs = dfset.make_dfset(filetype=filetype, csvsdir=MOCK_IN_DIR)

        sizes_df_orig, sizes_df_comp = sq.encode_measure_decode(
            dfs, codeclist)

    def _test_codecs_many_filetypes(self, codeclist, filetypes=None):
        if filetypes is None:
            filetypes = ('csv', 'npy', 'parquet', 'feather', 'h5')
        for ftype in filetypes:
            self._test_codecs_for_filetype(ftype, codeclist)

    def test_colsum(self):
        encs = [codec.ColSumPredictor(
            cols_to_sum='a', col_to_predict='b')]
        self._test_codecs_many_filetypes(encs)
        encs = [codec.Debug(), codec.ColSumPredictor(
            cols_to_sum='a', col_to_predict='b')]
        self._test_codecs_many_filetypes(encs)

    def _test_simple_codec(self, f_ctor, filetypes=None):
        encs = [f_ctor()]
        self._test_codecs_many_filetypes(encs, filetypes=filetypes)

        encs = [f_ctor(cols='a')]
        self._test_codecs_many_filetypes(encs, filetypes=filetypes)

        encs = [f_ctor(cols=['a', 'b'])]
        self._test_codecs_many_filetypes(encs, filetypes=filetypes)

        encs = [codec.Debug(), f_ctor()]
        self._test_codecs_many_filetypes(encs, filetypes=filetypes)

    def test_delta(self):
        # encs = [codec.Delta(cols='a')]
        # self._test_codecs_many_filetypes(encs, filetypes=['npy'])
        self._test_simple_codec(codec.Delta)

    def test_double_delta(self):
        self._test_simple_codec(codec.DoubleDelta)

    def test_dynamic_delta(self):
        self._test_simple_codec(codec.DynamicDelta)

        # self._test_simple_codec(codec.Zigzag)

    def test_quantize(self):
        # # encs = [codec.Quantize(cols='a')]
        # encs = [codec.Quantize(cols='c')]
        # # encs = [codec.Quantize()]
        # for ftype in ['npy']:  # csv drops dtype
        # for ftype in ('npy', 'parquet', 'feather', 'h5'):  # csv drops dtype
        #     self._test_codecs_for_filetype(ftype, encs)
        # self._test_codecs_many_filetypes(encs)

        def _test_array(a, compr_dtype=None):
            a = a.copy()
            enc = codec.Quantize()
            # print("a:\n", a)
            a_comp, qparams = enc.encode_col(a, 'foo')
            # print("qparams: ", qparams)
            # print("out:\n", out)
            a_hat = enc.decode_col(a_comp, 'foo', qparams)
            # print("a_hat:\n", a_hat)
            # assert np.array_equal(a, a_hat)
            assert a.dtype == a_hat.dtype
            if compr_dtype is not None:
                # print("a_comp.dtype:", a_comp.dtype)
                assert compr_dtype == a_comp.dtype
            # assert np.array_equal(a, a_hat, equal_nan=True)
            # assert np.allclose(a, a_hat, equal_nan=True)
            assert utils.allclose(a, a_hat, equal_nan=True)

        s = pd.Series([pd.NA, 0], dtype='Int8')  # nullable int
        _test_array(s, compr_dtype=np.uint8)
        # return  # TODO rm

        a = np.arange(4, dtype=np.float32)
        a[0] = np.nan
        _test_array(a, compr_dtype=np.uint8)
        a[-1] = np.nan
        _test_array(a, compr_dtype=np.uint8)
        a[:] = np.nan
        _test_array(a, compr_dtype=np.uint8)

        a = np.arange(4, dtype=np.int16)
        _test_array(a, compr_dtype=np.uint8)
        # return
        _test_array(a.astype(np.int32), compr_dtype=np.uint8)
        _test_array(a.astype(np.int64), compr_dtype=np.uint8)
        maxval = (1 << 32) + ((1 << 32) - 1)
        a = np.arange(maxval - 5, maxval + 1, dtype=np.uint64)
        _test_array(a, compr_dtype=np.uint8)
        a = np.array([0, maxval], dtype=np.uint64)
        _test_array(a)

        a = np.array([0, 255, np.nan], dtype=np.float32)
        _test_array(a, compr_dtype=np.uint16)  # since 255 and nan, needs u16
        a = np.array([0, 254, np.nan], dtype=np.float32)
        _test_array(a, compr_dtype=np.uint8)  # just 254 and nan, can do u8

        ftypes = ('npy', 'parquet', 'feather', 'h5')  # csv drops dtype
        self._test_simple_codec(codec.Quantize, filetypes=ftypes)

    def test_zigzag(self):
        ftypes = ('npy', 'parquet', 'feather', 'h5')  # csv drops dtype
        encs = [codec.Delta(cols='c'), codec.Zigzag(cols='c')]
        self._test_codecs_many_filetypes(encs, ftypes)
        encs = [codec.Quantize(cols='c'), codec.Delta(cols='c'), codec.Zigzag(cols='c')]
        self._test_codecs_many_filetypes(encs, ftypes)
        encs = [codec.Quantize(cols='a'), codec.Zigzag(cols='a')]
        self._test_codecs_many_filetypes(encs, ftypes)
        encs = [codec.Quantize(cols='c'), codec.Zigzag(cols='c')]
        self._test_codecs_many_filetypes(encs, ftypes)
        encs = [codec.Zigzag(cols='c')]
        self._test_codecs_many_filetypes(encs, ftypes)
        encs = [codec.Zigzag(cols=['a', 'c'])]
        self._test_codecs_many_filetypes(encs, ftypes)
        encs = [codec.Quantize(cols=['a', 'c']), codec.Zigzag(cols=['a', 'c'])]
        self._test_codecs_many_filetypes(encs, ftypes)

    def test_byteshuf(self):
        self._test_simple_codec(codec.ByteShuffle)

    def test_bzip2(self):
        # csv breaks bz2 bytestreams
        filetypes = ('npy', 'parquet', 'feather', 'h5')
        self._test_simple_codec(codec.Bzip2, filetypes=filetypes)

    def test_zstd(self):
        # csv breaks zstd bytestreams (I think?)
        filetypes = ('npy', 'parquet', 'feather', 'h5')
        self._test_simple_codec(codec.Zstd, filetypes=filetypes)

    def test_codecsearch(self):
        pipelines = [[codec.Delta(cols='a')]]
        encs = [codec.CodecSearch(pipelines=pipelines)]
        self._test_codecs_many_filetypes(encs)

        pipelines = [(codec.Delta(cols='a'), codec.Delta(cols='a'))]
        encs = [codec.CodecSearch(pipelines=pipelines)]
        self._test_codecs_many_filetypes(encs)

        pipelines += [[codec.Delta(cols='a')]]
        encs = [codec.CodecSearch(pipelines=pipelines)]
        self._test_codecs_many_filetypes(encs)

        pipelines += [[codec.DoubleDelta()]]
        encs = [codec.CodecSearch(pipelines=pipelines)]
        self._test_codecs_many_filetypes(encs)

        pipelines = [[codec.Delta(cols='c'), codec.Zigzag(cols='c')]]  # works
        encs = [codec.CodecSearch(pipelines=pipelines)]
        self._test_codecs_for_filetype('npy', encs)

        pipelines = [[codec.Quantize(cols='c'), codec.Delta(cols='c')]]  # works
        encs = [codec.CodecSearch(pipelines=pipelines)]
        self._test_codecs_for_filetype('npy', encs)

        pipelines = [[codec.Quantize(cols='c'), codec.Delta(cols='c'),
                      codec.Zigzag(cols='c')]]
        encs = [codec.CodecSearch(pipelines=pipelines)]
        self._test_codecs_for_filetype('npy', encs)


if __name__ == '__main__':
    unittest.main()
