#!/usr/bin/env python

from __future__ import absolute_import

import os
import numpy as np
import pandas as pd
import unittest

from python import dfset
from python import dfsqueeze as sq


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


def _populate_mock_input_dir(df0, df1):
    df0.to_csv(os.path.join(MOCK_IN_DIR, 'df0.csv'), index=False)
    df1.to_csv(os.path.join(MOCK_IN_DIR, 'df1.csv'), index=False)


def _rm_mock_input_files():
    os.path.remove(os.path.join(MOCK_IN_DIR, 'df0.csv'))
    os.path.remove(os.path.join(MOCK_IN_DIR, 'df1.csv'))


class TestDfSet(unittest.TestCase):

    def setUp(self):
        self.df0 = _debug_df0()
        self.df1 = _debug_df1()
        _populate_mock_input_dir(self.df0, self.df1)

    def teatDown(self):
        _rm_mock_input_files()

    def _test_dfs(self, filetype):
        dfs = dfset.make_dfset(filetype=filetype, csvsdir=MOCK_IN_DIR)
        # dfs = dfset.make_dfset(filetype='csv')
        # print("dfs._dfsdir", dfs._dfsdir)
        # print("os.listdir(self._dfsdir)", os.listdir(dfs._dfsdir))
        # dfs.copy_from_csvs_dir(MOCK_IN_DIR)
        # print("os.listdir(self._dfsdir)", os.listdir(dfs._dfsdir))
        # print("dfs._dfsdir", dfs._dfsdir)
        # print("dfs.ids():", dfs.ids())
        # print("dfs._find_ids():", dfs._find_ids())

        assert sorted(dfs.ids()) == ['df0', 'df1']
        # print(dfs._cols_stored_for_dfid('df0'))
        assert sorted(dfs._cols_stored_for_dfid('df0')) == 'a b c'.split()
        assert sorted(dfs._cols_stored_for_dfid('df1')) == 'a b d'.split()

        df0_hat = dfs['df0']
        df1_hat = dfs['df1']
        assert self.df0.shape == df0_hat.shape
        # assert self.df0.dtypes == df0_hat.dtypes  # not true for csv
        # print(self.df0.dtypes)
        # print(df0_hat.dtypes)
        assert set(self.df0.columns) == set(df0_hat.columns)
        assert set(self.df1.columns) == set(df1_hat.columns)
        for col in self.df0:
            assert np.allclose(self.df0[col], df0_hat[col])
        for col in self.df1:
            assert np.allclose(self.df1[col], df1_hat[col])
        # assert np.allclose(self.df0['a'], df0_hat['a'])
        # assert np.allclose(self.df0['b'], df0_hat['b'])
        # assert np.allclose(self.df0['c'], df0_hat['c'])
        # assert self.df0.equals(df0_hat)  # csv discards dtypes
        # assert self.df1.equals(dfs['df1'])  # csv discards dtypes

    def test_csv_dfs(self):
        self._test_dfs(filetype='csv')

    def test_npy_dfs(self):
        pass

    def test_parquet_dfs(self):
        pass

    def test_h5_dfs(self):
        pass


if __name__ == '__main__':
    _populate_mock_input_dir()
    unittest.main()
