#!/usr/bin/env python

import numpy as np
import pandas as pd


# we need these classes because different columns in a dataframe might
# up getting compressed different amounts / generally no longer act
# like 1d arrays, which makes pandas dataframes unhappy; in particular, if
# you start with two columns that have N elements, and then compress the first
# to B != kN bytes for some int k, there's no longer a way to store both of
# these columns in the same dataframe (at least not without terrible
# workarounds)


# this just exists so calling df[col].values works like with a regular
# pandas df
class SimpleSeries(object):
    __slots__ = 'values dtype'.split()

    def __init__(self, vals):
        try:
            self.values = np.asarray(vals.values)  # handle pd series
        except AttributeError:
            self.values = np.asarray(vals)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return self.values[idx]

    def __setitem__(self, idx, val):
        self.values[idx] = val

    def __iter__(self):
        return self.values.__iter__()

    def __str__(self):
        return str(self.values)


class SimpleDataFrame(object):
    __slots__ = '_col2array'

    def __init__(self, df=None):
        self._col2array = {}
        if df is not None:
            for col in df:
                self[col] = df[col]

    @staticmethod
    def from_pandas_df(df):
        ret = SimpleDataFrame()
        for col in df:
            ret[col] = df[col]
        return ret

    @staticmethod
    def from_dict(d):
        df = SimpleDataFrame()
        for key in d:
            df[key] = d[key]
        return df

    @property
    def columns(self):
        # return sorted(list(self._col2array.keys()))
        return list(self._col2array.keys())

    @property
    def shape(self):
        ncols = len(self.columns)
        if ncols < 1:
            return (0, 0)
        lengths = [len(ar) for ar in self._col2array.values()]
        return (np.max(lengths), ncols)

    @property
    def dtypes(self):
        # return {col: self._col2array[col].values.dtype
        #         for col in self._col2array.keys()}
        return SimpleSeries([self._col2array[col].values.dtype
                             for col in self.columns])

    def __getitem__(self, col_or_cols):
        if isinstance(col_or_cols, list):
            ret = SimpleDataFrame()
            for col in col_or_cols:
                ret[col] = self[col]
            return ret
        return self._col2array[col_or_cols]

    def __setitem__(self, col, vals):
        self._col2array[col] = SimpleSeries(vals)

    def __iter__(self):
        return SimpleDataFrameIterator(self)

    def __len__(self):
        return len(self.columns)



class SimpleDataFrameIterator(object):
    __slots__ = '_idx _cols'.split()

    def __init__(self, df):
        # self._df = df
        self._idx = 0
        self._cols = df.columns

    def __next__(self):
        try:
            ret = self._cols[self._idx]
            self._idx += 1
            return ret
        except IndexError:
            raise StopIteration()

    # def __len__(self):
    #     return len(self._cols)
