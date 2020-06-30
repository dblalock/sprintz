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


def _populate_mock_input_dir(df0, df1):
    df0.to_csv(os.path.join(MOCK_IN_DIR, 'df0.csv'), index=False)
    df1.to_csv(os.path.join(MOCK_IN_DIR, 'df1.csv'), index=False)


def _rm_mock_input_files():
    os.remove(os.path.join(MOCK_IN_DIR, 'df0.csv'))
    os.remove(os.path.join(MOCK_IN_DIR, 'df1.csv'))


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
        for col in self.df1:
            assert np.array_equal(self.df1[col], df1_hat[col])

        # assert np.allclose(self.df0['a'], df0_hat['a'])
        # assert np.allclose(self.df0['b'], df0_hat['b'])
        # assert np.allclose(self.df0['c'], df0_hat['c'])
        # assert np.allclose(self.df1['a'], df1_hat['a'])
        # assert np.allclose(self.df1['b'], df1_hat['b'])
        # assert np.allclose(self.df1['d'], df1_hat['d'])
        # assert np.array_equal(self.df1['a'], df1_hat['a'])
        # assert np.array_equal(self.df1['b'], df1_hat['b'])
        # assert np.array_equal(self.df1['d'], df1_hat['d'])
        # if filetype != 'csv':
            # print(self.df1.dtypes)
            # print(df1_hat.dtypes)
            # # print(df1_hat.columns)
            # # print(self.df1.columns)
            # print(self.df1['a'])
            # print(df1_hat['a'])

            # this function doesn't seem reliable; returns false for two
            # dfs with same cols, dtypes, and all values pass allclose;
            # maybe related to weird warnings about ufunc size changing
            # and latest numpy being broken for me
            # assert self.df0.equals(df0_hat)
            # assert self.df1.equals(df1_hat)

    def test_csv_dfs(self):
        self._test_dfs(filetype='csv')

    def test_npy_dfs(self):
        self._test_dfs(filetype='npy')

    def test_parquet_dfs(self):
        self._test_dfs(filetype='parquet')

    def test_h5_dfs(self):
        self._test_dfs(filetype='h5')


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
            filetypes = ('csv', 'npy', 'parquet', 'h5')
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
        # for ftype in ['npy']:  # csv doesn't preserve dtype
        # for ftype in ('npy', 'parquet', 'h5'):  # csv doesn't preserve dtype
        #     self._test_codecs_for_filetype(ftype, encs)
        # self._test_codecs_many_filetypes(encs)

        ftypes = ('npy', 'parquet', 'h5')  # csv doesn't preserve dtype
        self._test_simple_codec(codec.Quantize, filetypes=ftypes)

    def test_zigzag(self):
        ftypes = ('npy', 'parquet', 'h5')  # csv doesn't preserve dtype
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

    def test_codecsearch(self):
        # pipelines = [[codec.Delta(cols='a')]]
        # encs = [codec.CodecSearch(pipelines=pipelines)]
        # self._test_codecs_many_filetypes(encs)

        # pipelines = [(codec.Delta(cols='a'), codec.Delta(cols='a'))]
        # encs = [codec.CodecSearch(pipelines=pipelines)]
        # self._test_codecs_many_filetypes(encs)

        # pipelines += [[codec.Delta(cols='a')]]
        # encs = [codec.CodecSearch(pipelines=pipelines)]
        # self._test_codecs_many_filetypes(encs)

        # pipelines += [[codec.DoubleDelta()]]
        # encs = [codec.CodecSearch(pipelines=pipelines)]
        # self._test_codecs_many_filetypes(encs)

        # pipelines += [[codec.Quantize(), codec.DoubleDelta()]]
        # pipelines = [[codec.Quantize(), codec.Delta()]]
        # pipelines = [[codec.Delta()]]
        # pipelines = [[codec.Quantize(cols='a'), codec.Delta(cols='a')]]
        # pipelines = [[codec.Quantize(cols='a'), codec.DoubleDelta(cols='a')]]
        # pipelines = [[codec.Quantize()]]
        # pipelines = [[codec.Quantize(cols='a')]]
        # pipelines = [[codec.Delta(cols='a')]]
        # pipelines = [[codec.Quantize(cols='a'), codec.Delta(cols='a')]]
        # pipelines += [[codec.Quantize(cols='c'), codec.Delta(cols='c')]]
        # pipelines += [[codec.Quantize(cols='c'), codec.DoubleDelta(cols='c')]]
        # pipelines += [[codec.Quantize(cols='c'), codec.Delta(cols='c'),
        pipelines = [[codec.Quantize(cols='c'), codec.Delta(cols='c'),
                      codec.Zigzag(cols='c')]]
        encs = [codec.CodecSearch(pipelines=pipelines)]
        self._test_codecs_for_filetype('npy', encs)
        # ftypes = ('npy', 'parquet', 'h5')  # csv doesn't preserve dtype
        # self._test_codecs_many_filetypes(encs, filetypes=ftypes)



if __name__ == '__main__':
    unittest.main()
