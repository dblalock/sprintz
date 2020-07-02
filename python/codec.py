#!/usr/bin/env python

import abc
# import copy
import bz2

import numpy as np
from scipy import signal
import numba

from python import dfquantize2 as dfq
from python import learning  # for compute_loss
from python import compress


def _wrap_in_list_if_str(obj):
    return ([obj] if isinstance(obj, str) else obj)


class BaseCodec(abc.ABC):

    def __init__(self, cols=None):
        self._cols = _wrap_in_list_if_str(cols)  # None = all
        self._needs_training = False

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
        return self._cols
        # return None  # None = all of them

    # @property
    def train_cols(self):
        return self._cols

    def train(self, dfc):
        pass

    def cols_to_use(self, df):
        cols = self.cols()
        if cols is not None:
            return sorted(set(cols) & set(df.columns))
        return sorted(df.columns)

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
        use_cols = self.cols_to_use(df)
        # print("basecodec use cols: ", use_cols)
        col2header = {}
        for col in use_cols:
            df[col], header = self.encode_col(df[col].values, col)
            if header is not None:
                col2header[col] = header
        return df[use_cols], col2header or None  # no headers -> None

    def decode(self, df, header):
        col2header = header or {}
        use_cols = self.cols_to_use(df)
        for col in use_cols:
            df[col] = self.decode_col(
                df[col].values, col, col2header.get(col))
        return df[use_cols]

    def encode_col(self, values, col):
        pass  # either overwrite this or encode()

    def decode_col(self, values, col, header):
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


@numba.njit(fastmath=True)
def _cumsum_1d(x):
    n = len(x)
    out = np.empty(n, x.dtype)
    out[0] = x[0]
    for j in range(1, n):
        out[j] = out[j - 1] + x[j]
    return out


class Delta(BaseCodec):

    def encode_col(self, vals, col_unused):
        vals[1:] -= vals[:-1]
        return vals, None

    def decode_col(self, vals, col_unused, header_unused):
        return _cumsum_1d(vals)


class DoubleDelta(BaseCodec):

    def encode_col(self, vals, col_unused):
        vals[1:] -= vals[:-1]
        vals[1:] -= vals[:-1]
        return vals, None

    def decode_col(self, vals, col_unused, header_unused):
        return _cumsum_1d(_cumsum_1d(vals))


class DynamicDelta(BaseCodec):

    def __init__(self, *args, block_len=3, loss='logabs', **kwargs):
        super().__init__(*args, **kwargs)
        self._block_len = block_len
        assert loss in ('l1', 'l2', 'linf', 'logabs')
        self._loss = loss

    def encode(self, df):
        use_cols = self.cols_to_use(df)
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
        use_cols = self.cols_to_use(df)
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

    def encode_col(self, vals, col_unused):
        if vals.itemsize == 1:
            return vals  # shuffling is no-op for 1B dtypes
        X = vals.view(np.uint8).reshape(-1, vals.itemsize)
        return np.asfortranarray(X).ravel().view(vals.dtype), None

    def decode_col(self, vals, col_unused, header_unused):
        if vals.itemsize == 1:
            return vals  # shuffling is no-op for 1B dtypes
        X = vals.view(np.uint8).reshape(vals.itemsize, -1)
        return np.asfortranarray(X).ravel().view(vals.dtype)


class CodecSearch(BaseCodec):

    def __init__(self, pipelines, loss='zstd', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pipelines = pipelines
        self._loss = loss

    def encode(self, df):
        col2headers = {}
        use_cols = self.cols_to_use(df)
        # print("codecsearch cols to use: ", use_cols)
        for col in use_cols:
            df[col], (pipeline_idx, headers) = self.encode_col(
                df[col].values, col)
            if len(headers) and any(headers):
                col2headers[col] = (pipeline_idx, headers)
            else:
                col2headers[col] = pipeline_idx
        return df[use_cols], col2headers

    def encode_col(self, vals, col):
        orig_vals = vals
        best_loss = np.inf
        best_pipeline_idx = None
        best_vals = None
        best_headers = None
        for i, pipeline in enumerate(self._pipelines):
            vals = orig_vals.copy()
            headers = []
            for enc in pipeline:

                # skip codecs that don't think they're supposed to run on
                # this columns
                enc_cols = enc.write_cols()
                # print("encode: col, enc_cols, cls = ", col, enc_cols, enc.__class__)
                if enc_cols is not None and col not in enc_cols:
                    continue
                # print("not skipping!")

                vals, header = enc.encode_col(vals, col)

                # if col == 'c':
                #     print(f"enc_col {col}, {type(enc)}: got vals:\n", vals, vals.dtype)

                headers.append(header)
            # print(f"enc col {col}, headers={headers} list with type {type(headers[0])}")
            loss = learning.compute_loss(vals, loss=self._loss)
            if loss < best_loss:
                # print("got new best loss: ", loss)
                best_loss = loss
                best_pipeline_idx = i
                best_vals = vals
                best_headers = headers
        # print(f"enc col {col}, headers={best_headers} list with type {type(best_headers[0])}")
        # combined_header = (best_pipeline_idx, best_headers)
        # print(f"enc col {col}, type of returned header, header[1] = {type(combined_header)}, {type(combined_header[1])}")
        # if col == 'c':
        #     print(f"enc_col {col}: returning vals:\n", best_vals, best_vals.dtype)
        return best_vals, (best_pipeline_idx, best_headers)

    def decode_col(self, vals, col, header):
        # print("search decoding col, header", col, header)
        # print("type of header: ", type(header))
        if isinstance(header, tuple):
            assert len(header) == 2  # should have pipeline idx and headerlist
            pipeline_idx, headers_list = header
            # print("pipeline idx:", pipeline_idx)
            pipeline = self._pipelines[pipeline_idx]
            headers_list = headers_list
        else:
            pipeline_idx = header
            pipeline = self._pipelines[pipeline_idx]
            headers_list = [None for enc in pipeline]

        for i in range(len(pipeline))[::-1]:
            enc = pipeline[i]
            header = headers_list[i]

            enc_cols = enc.write_cols()
            # print("decode: col, enc_cols, cls = ", col, enc_cols, enc.__class__)
            if enc_cols is not None and col not in enc_cols:
                continue
            # print("not skipping!")

            vals = enc.decode_col(vals, col, header)
            # if col == 'c':
            #     print(f"dec_col {col}, {type(enc)}: got vals:\n", vals, vals.dtype)
        return vals

    def decode(self, df, headers):
        col2headers = headers
        use_cols = self.cols_to_use(df)
        for col in use_cols:
            df[col] = self.decode_col(df[col].values, col, col2headers[col])
        return df[use_cols]


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
        predictions = df[self.cols_to_sum[0]].values
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
        df[self.col_to_predict] = vals - predictions.astype(vals.dtype)
        # NOTE: have to give it a list of cols or it returns a series, not df
        return df[[self.col_to_predict]], None  # no headers

    def decode(self, df, header_unused):
        predictions = self._compute_predictions(df)
        # df[self.col_to_predict] = df[self.col_to_predict].values + predictions
        # NOTE: have to give it a list of cols or it returns a series, not df
        vals = df[self.col_to_predict].values
        # df.drop(self.col_to_predict, axis=1, inplace=True)
        df[self.col_to_predict] = vals + predictions.astype(vals.dtype)
        return df[[self.col_to_predict]]


class Quantize(BaseCodec):

    def __init__(self, *args, col2qparams=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._col2qparams = col2qparams or {}

        # TODO support just going from f64 to f32 (as opposed outputing ints)?

    def encode_col(self, vals, col):
        # print("quantize encoding col", col)
        if col in self._col2qparams:
            qparams = self._col2qparams[col]
            return_qparams = False
        else:
            # infered params need to be saved as headers
            qparams = dfq.infer_qparams(vals)
            return_qparams = True

        ret = dfq.quantize(vals, qparams)
        # print("returning quantized vals:\n", ret, ret.dtype)
        assert ret.dtype == qparams.dtype
        return ret, qparams if return_qparams else None

    def decode_col(self, vals, col, qparams):
        if qparams is None:
            qparams = self._col2qparams[col]  # None because defined a priori
        # print("quantize decoding col using qparams", col, qparams)
        return dfq.unquantize(vals, qparams)


class Zigzag(BaseCodec):

    def encode_col(self, vals, col_unused):
        return compress.zigzag_encode(vals), None

    def decode_col(self, vals, col_unused, header_unused):
        # print(f"zigzag decoding col: {col_unused} with dtype {vals.dtype}")
        # print("zigzag dec got vals: ", vals, vals.dtype)
        return compress.zigzag_decode(vals)


class Bzip2(BaseCodec):

    # so the non-obvious thing here is that we need to stop all the other
    # code from breaking by ensuring that what we return (both from encode
    # *and* decode) is a numpy array; and since type info is lost, we
    # need to save it as a header

    def encode_col(self, vals, col_unused):
        ret = bz2.compress(vals)
        # print("bz2 enc ret type: ", type(ret))
        ret = np.frombuffer(ret, dtype=np.uint8)
        # print("bz2 enc ret type: ", type(ret))
        # print("bz2 enc ret type: ", type(ret), ret.dtype)
        return ret, vals.dtype
        # return bz2.compress(vals), None

    def decode_col(self, vals, col_unused, header):
        orig_dtype = header
        # print("bz2 dec vals type, dtype: ", type(vals), vals.dtype)
        ret = bz2.decompress(vals.tobytes())
        ret = np.frombuffer(ret, dtype=orig_dtype)
        # print("bz2 ret type: ", type(ret))
        # print("bz2 ret type: ", type(ret), ret.dtype)
        return ret


class Zstd(BaseCodec):

    def encode_col(self, vals, col_unused):
        ret = compress.zstd_compress(vals)
        ret = np.frombuffer(ret, dtype=np.uint8)
        return ret, vals.dtype

    def decode_col(self, vals, col_unused, header):
        orig_dtype = header
        ret = compress.zstd_decompress(vals)
        ret = np.frombuffer(ret, dtype=orig_dtype)
        return ret
