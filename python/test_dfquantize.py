#!/usr/bin/env python

from __future__ import absolute_import

import os
import numpy as np
import pandas as pd
import unittest

from python import dfquantize as dfq

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


def _populate_mock_input_dir():
    df0 = _debug_df0()
    df1 = _debug_df1()
    df0.to_csv(os.path.join(MOCK_IN_DIR, 'df0.csv'), index=False)
    df1.to_csv(os.path.join(MOCK_IN_DIR, 'df1.csv'), index=False)


class TestDfQuantize(unittest.TestCase):

    def setUp(self):
        pass

    def test_ndigits_stats(self):
        assert dfq.ndigits_before_after_decimal('123') == (3, 0)
        assert dfq.ndigits_before_after_decimal('12.') == (2, 0)
        assert dfq.ndigits_before_after_decimal('.12') == (0, 2)
        assert dfq.ndigits_before_after_decimal('1.2') == (1, 1)
        assert dfq.ndigits_before_after_decimal('09.04') == (1, 2)
        assert dfq.ndigits_before_after_decimal('090.0400') == (2, 2)
        assert dfq.ndigits_before_after_decimal('00.000') == (0, 0)
        assert dfq.ndigits_before_after_decimal('00000') == (0, 0)
        assert dfq.ndigits_before_after_decimal('0') == (0, 0)
        assert dfq.ndigits_before_after_decimal('.') == (0, 0)
        assert dfq.ndigits_before_after_decimal('') == (0, 0)

    def test_quantize_params(self):
        # a = np.array([0, 0, 0])
        # pass
        _populate_mock_input_dir()

    def test_quantize_dfs(self):
        pass
        # return
        # _populate_mock_input_dir()
        # dfq.quantize_dfs(MOCK_IN_DIR, MOCK_OUT_DIR, MOCK_CONTEXT_PATH)
        # for df, name in [(_debug_df0(), 'df0'), (_debug_df1(), 'df1')]:
        #     out_path = os.path.join(MOCK_OUT_DIR, name + '.dat')
        #     assert os.path.exists(out_path)
        #     dfhat = pd.read_csv(out_path)
        #     print("df, dfhat")
        #     print(df)
        #     print(dfhat)
        #     # assert False


if __name__ == '__main__':
    _populate_mock_input_dir()
    unittest.main()
