#!/usr/bin/env python

import abc
import copy

import numpy as np
from scipy import signal

from python import dfquantize2 as dfq
from python import learning  # for compute_loss


def _wrap_in_list_if_str(obj):
    return ([obj] if isinstance(obj, str) else obj)


class BaseCodec(abc.ABC):

    def __init__(self, cols=None):
        self._needs_training = False
        self._cols = _wrap_in_list_if_str(cols)  # None = all

    # @property
    def cols(self):
        readonly_cols = self.readonly_cols() or []
        write_cols = self.write_cols() or []
        return (readonly_cols + write_cols) or None

    # @property
    def readonly_cols(self):
        return None  # None = all of them

    # @abc.abstractmethod
    # @property
    def write_cols(self):
        return None  # None = all of them

    # @property
    def train_cols(self):
        return self._cols

    def train(self, dfc):
        pass

    def _cols_to_use(self, df):
        return self._cols if self._cols is not None else df.columns

    # @property
    def needs_training(self):
        return self._needs_training

    # @needs_training.setter
    # def needs_training(self, val):
    #     self._needs_training = val

    # TODO using json-serializable params would be less brittle
    # than pickling in case the class definition changes between
    # serialization and deserialization
    #
    # # @abc.abstractmethod
    # # # @property
    # def params(self):
    #     pass
    #
    # @classmethod
    # def from_params(self, params):
    #     pass

    def encode(self, df):
        use_cols = self._cols_to_use(df)
        for col in use_cols:
            df[col] = self.encode_col(df[col].values)
        return df[use_cols], None

    def decode(self, df, header):
        use_cols = self._cols_to_use(df)
        for col in use_cols:
            df[col] = self.decode_col(df[col].values)
        return df[use_cols]

    def encode_col(self, values):
        pass  # either overwrite this or encode()

    def decode_col(self, values):
        pass  # either overwrite this or decode()


class Debug(BaseCodec):

    def encode(self, df):
        # return df.columns, None # TODO rm after debug
        for col in df:
            df[col] = df[col].values[::-1]
            # df[col] -= 1
        return df, None

    def decode(self, df, header_unused):
        # return # TODO rm after debug
        for col in df:
            df[col] = df[col].values[::-1]
            # df[col] += 1
        # print("df col a within decode: ", df['a'])
        return df


class Delta(BaseCodec):

    def encode_col(self, vals):
        vals[1:] -= vals[:-1]
        return vals

    def decode_col(self, vals):
        return np.cumsum(vals)

    # def encode(self, df):
    #     # print("running encode!")
    #     # return df.columns, None # TODO rm after debug
    #     # use_cols = self._cols if self._cols is not None else df.columns
    #     use_cols = self._cols_to_use(df)
    #     for col in use_cols:
    #         vals = df[col].values
    #         vals[1:] -= vals[:-1]
    #         df[col] = vals
    #         # df[col] = df[col].values[::-1]  # TODO rm after debug
    #     return df[use_cols], None

    # def decode(self, df, header_unused):
    #     # print("running decode!")
    #     # return # TODO rm after debug
    #     # use_cols = self._cols if self._cols is not None else df.columns
    #     use_cols = self._cols_to_use(df)
    #     for col in use_cols:
    #         df[col] = np.cumsum(df[col])
    #         # df[col] = df[col].values[::-1]  # TODO rm after debug
    #     return df[use_cols]


class DoubleDelta(BaseCodec):

    def encode_col(self, vals):
        vals[1:] -= vals[:-1]
        vals[1:] -= vals[:-1]
        return vals

    def decode_col(self, vals):
        return np.cumsum(np.cumsum(vals))

    # def encode(self, df):
    #     # print("running encode!")
    #     # return df.columns, None # TODO rm after debug
    #     # use_cols = self._cols if self._cols is not None else df.columns
    #     use_cols = self._cols_to_use(df)
    #     for col in use_cols:
    #         vals = df[col].values
    #         vals[1:] -= vals[:-1]
    #         vals[1:] -= vals[:-1]
    #         df[col] = vals
    #     return df[use_cols], None

    # def decode(self, df, header_unused):
    #     # print("running decode!")
    #     # return # TODO rm after debug
    #     # use_cols = self._cols if self._cols is not None else df.columns
    #     use_cols = self._cols_to_use(df)
    #     for col in use_cols:
    #         df[col] = np.cumsum(np.cumsum(df[col]))
    #     return df[use_cols]


class DynamicDelta(BaseCodec):

    def __init__(self, *args, block_len=3, loss='logabs', **kwargs):
        super().__init__(*args, **kwargs)
        self._block_len = block_len
        assert loss in ('l1', 'l2', 'linf', 'logabs')
        self._loss = loss

    def encode(self, df):
        use_cols = self._cols_to_use(df)
        trailing_len = df.shape[0] % self._block_len
        # n = df.shape[0]
        # nblocks = n // self._block_len
        if df.shape[0] < self._block_len:  # no blocks
            return df[use_cols], None

        col2mask = {}
        for col in use_cols:
            # construct delta coded and double-delta coded versions
            vals_delta = df[col].values
            vals_delta[1:] -= vals_delta[:-1]
            vals_double = vals_delta.copy()
            vals_double[1:] -= vals_double[:-1]

            orig_vals_delta = vals_delta.copy()
            if trailing_len > 0:
                vals_delta = vals_delta[:-trailing_len]
                vals_double = vals_double[:-trailing_len]
            X_delta = vals_delta.reshape(-1, self._block_len)
            X_double = vals_double.reshape(-1, self._block_len)

            losses_delta = learning.compute_loss(X_delta, loss=self._loss)
            losses_double = learning.compute_loss(X_delta, loss=self._loss)
            # X_out = X_delta.copy()
            X_out = X_delta
            mask = losses_double < losses_delta
            X_out[mask] = X_double
            col2mask[col] = np.packbits(mask)

            ret = orig_vals_delta
            if trailing_len > 0:
                ret[:-trailing_len] = X_out.ravel()
            else:
                ret = X_out.ravel()
            df[col] = ret

        return df[use_cols], col2mask

    def decode(self, df, header):
        use_cols = self._cols_to_use(df)
        trailing_len = df.shape[0] % self._block_len
        nblocks = df.shape[0] // self._block_len
        if nblocks < 1:  # no blocks
            return df[use_cols]

        col2mask = header
        for col in use_cols:
            vals_delta = np.cumsum(df[col].values)
            vals_double = np.cumsum(vals_delta)

            orig_vals_delta = vals_delta.copy()
            if trailing_len > 0:
                vals_delta = vals_delta[:-trailing_len]
                vals_double = vals_double[:-trailing_len]

            X_delta = vals_delta.reshape(-1, self._block_len)
            X_double = vals_double.reshape(-1, self._block_len)
            mask = np.unpackbits(col2mask[col], count=nblocks).astype(np.bool)
            X_out = X_delta
            X_out[mask] = X_double

            ret = orig_vals_delta
            if trailing_len > 0:
                ret[:-trailing_len] = X_out.ravel()
            else:
                ret = X_out.ravel()
            df[col] = ret

        return df[use_cols]


class ByteShuffle(BaseCodec):

    def encode_col(self, vals):
        if vals.itemsize == 1:
            return vals  # shuffling is no-op for 1B dtypes
        X = vals.view(np.uint8).reshape(-1, vals.itemsize)
        return np.asfortranarray(X).ravel().view(vals.dtype)

    def decode_col(self, vals):
        if vals.itemsize == 1:
            return vals  # shuffling is no-op for 1B dtypes
        X = vals.view(np.uint8).reshape(vals.itemsize, -1)
        return np.asfortranarray(X).ravel().view(vals.dtype)


class CodecSearch(BaseCodec):

    def __init__(self, pipelines, loss='logabs', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pipelines = pipelines
        self._loss = loss

    def encode_col(self, vals):
        orig_vals = vals.copy()
        for pipeline in self._pipelines:
            pass # TODO


class ColSumPredictor(BaseCodec):
    """predicts a column as the (weighted) sum of one or more other columns"""

    def __init__(self, cols_to_sum, col_to_predict, weights=None):
        super().__init__()
        self.cols_to_sum = _wrap_in_list_if_str(cols_to_sum)
        self.col_to_predict = col_to_predict
        self._weights = weights
        # if weights is not None and weights != 'auto':
        if self._weights is not None:  # TODO support regression to infer weights
            # assert weights.ndim == 2  #
            assert self._weights.shape[-1] == len(self.cols_to_sum)
            if len(cols_to_sum) == 1:
                self._weights = self._weights.reshape(-1, 1)

    def readonly_cols(self):
        return self.cols_to_sum

    def write_cols(self):
        return [self.col_to_predict]

    def _compute_predictions(self, df):
        predictions = df[self.cols_to_sum[0]]
        if self._weights is not None:
            predictions = signal.correlate(
                predictions, self._weights[:, 0], padding='valid')
        if len(self.cols_to_sum) > 1:
            for i, col in enumerate(self.cols_to_sum[1:]):
                # predictions += df[col].values
                vals = df[col].values
                if self._weights is not None:
                    vals = signal.correlate(
                        vals, self._weights[:, i + 1], padding='valid')
                predictions += vals
        return predictions

    def encode(self, df):
        predictions = self._compute_predictions(df)
        vals = df[self.col_to_predict].values
        # df.drop(self.col_to_predict, axis=1, inplace=True)
        df[self.col_to_predict] = vals - predictions
        # NOTE: have to give it a list of cols or it returns a series, not df
        return df[[self.col_to_predict]], None  # no headers

    def decode(self, df, header_unused):
        predictions = self._compute_predictions(df)
        # df[self.col_to_predict] = df[self.col_to_predict].values + predictions
        # NOTE: have to give it a list of cols or it returns a series, not df
        vals = df[self.col_to_predict].values
        # df.drop(self.col_to_predict, axis=1, inplace=True)
        df[self.col_to_predict] = vals + predictions
        return df[[self.col_to_predict]]


class Quantize(BaseCodec):

    def __init__(self, *args, col2qparams=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._needs_training = True
        self._col2qparams = col2qparams or {}

        # TODO support just going from f64 to f32 (as opposed outputing ints)?

    def encode(self, df):
        use_cols = self._cols_to_use(df)
        col2qparams = {}
        for col in use_cols:
            vals = df[col].values
            if col in self._col2qparams:
                qparams = self._col2qparams[col]
            else:
                # infered params need to be saved as headers
                qparams = dfq.infer_qparams(vals)
                col2qparams[col] = qparams

            # import pprint
            # print("col: ", col)
            # pprint.pprint(col2qparams)
            # print("df dtypes:\n", df.dtypes)
            # print("df[col]:\n", df[col])

            df[col] = dfq.quantize(vals, qparams)
            assert df.dtypes[col] == qparams.dtype

            # vals = df[col].to_numeric().values
            # print("vals dtype:", vals.dtype)
            # print("vals dtype:", df[col].dtype)
            # import pandas as pd
            # print(df[col])
            # print("df dtypes before numeric:\n", df.dtypes)
            # df[col] = pd.to_numeric(df[col])
            # print("df dtypes after numeric:\n", df.dtypes)
            # df[col] = pd.to_numeric(df[col]).astype(np.int)
            # df.dtypes[col] = qparams.dtype
            # df.dtypes[col] = qparams.dtype
            # print("col: ", col)
            # print("vals: ", vals)
            # print("df dtypes after quantize:\n", df.dtypes)
            # print("qparams dtype: ", qparams.dtype)

        return df[use_cols], col2qparams

    def decode(self, df, col2qparams):
        use_cols = self._cols_to_use(df)
        if self._col2qparams:
            # header only contains info for cols not specified a priori
            col2qparams = copy.deepcopy(col2qparams)
            col2qparams.update(self._col2qparams)
        for col in use_cols:
            vals = df[col].values
            qparams = col2qparams[col]
            df[col] = dfq.unquantize(vals, qparams)
        return df[use_cols]
