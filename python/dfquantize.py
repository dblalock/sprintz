#!/usr/bin/env python

import collections
import os
import numpy as np
import numba
import pandas as pd

from pprint import pprint # TODO rm


class QuantizeParams(object):
    __slots__ = 'dtype scale'.split()


class Variable(object):
    __slots__ = 'name dtype offset'.split()


class Schema(object):
    __slots__ = 'variables'

# import pathlib as pl

# from . import compress


# @numba.njit(fastmath=True, cache=True)
# def ndigits_after_decimal(string):
#     decimal_idx = string.find('.') % len(string)  # treat -1 and end the same
#     return len(string) - decimal_idx - 1


# @numba.njit(fastmath=True, cache=True)
# def ndigits_before_decimal(string):
#     decimal_idx = string.find('.')
#     if decimal_idx == -1:
#         return len(str)
#     return decimal_idx


# @numba.njit(fastmath=True, cache=True)
def _ndigits_before_after_decimal(string):
    # i = 0
    # while (string[i] == '0'):
    #     i += 1
    # string = string[i:]
    string = string.strip('0')
    decimal_idx = string.find('.')
    n = len(string)
    if decimal_idx == -1:
        return n, 0
    return decimal_idx, n - decimal_idx - 1


def ndigits_before_after_decimal(string_or_strings):
    if not isinstance(string_or_strings, str):
        stats = [_ndigits_before_after_decimal(s) for s in string_or_strings]
        before, after = zip(*stats)
        return np.max(before), np.max(after)
    return _ndigits_before_after_decimal(string_or_strings)


def _generator_for_dfs_dir(dirpath, endswith='.csv', **read_csv_kwargs):

    def dfs_gen(dirpath):
        fnames = os.listdir(dirpath)
        if endswith is not None:
            fnames = [fname for fname in fnames if fname.endswith(endswith)]
        for fname in fnames:
            dfpath = os.path.join(dirpath, fname)
            yield pd.read_csv(dfpath, **read_csv_kwargs)

    return dfs_gen(dirpath)


# def col_digit_stats(dfs_or_dfs_dir, **generator_kwargs):
# def col_digit_stats(dfs, subtract_val='min'):
def col_digit_stats(dfs, subtract_val=None):

    # # if got a directory of dfs, wrap them in a generator for convenience
    # if isinstance(dfs_or_dfs_dir, (str, pl.Path)):
    #     return col_digit_stats(_generator_for_dfs_dir(**generator_kwargs))

    col2digitstats = {}
    for df in dfs:
        for col in df.columns.values:
            vals = df[col]
            if subtract_val is not None:
                vals = vals.values.astype(np.float64)
                if subtract_val == 'min':
                    subtract_val = vals.min()
                vals -= subtract_val
                # TODO going back to string and using string repr to infer
                # number of digits needed is inefficient
                vals = np.array([str(val) for val in vals])
            before, after = ndigits_before_after_decimal(vals)
            cur_before, cur_after = col2digitstats.get(col, (0, 0))
            col2digitstats[col] = (max(cur_before, before),
                                   max(cur_after, after))
    return col2digitstats


def quantization_dtype_scale(ndigits_before_decimal, ndigits_after_decimal):
    scale = int(10 ** ndigits_after_decimal)
    maxval_possible = 10 ** (ndigits_before_decimal + ndigits_after_decimal)
    nbits = int(np.ceil(np.log2(maxval_possible)))
    if nbits <= 16:
        dtype = np.uint16
    elif nbits <= 32:
        dtype = np.uint32
    else:
        dtype = np.uint64
    return dtype, scale


def _quantize_df(df, col2qparams):
    new_cols = {}
    minvals = {}
    for col in df.columns.values:
        dtype, scale = col2qparams[col]
        assert df[col].isna().sum() == 0   # fail fast if there are nans
        quantized = df[col].values.astype(np.float64) * scale
        minval = quantized.min()
        quantized -= minval
        new_cols[col] = quantized.astype(dtype)
        minvals[col] = minval
    return pd.DataFrame.from_dict(new_cols), minvals


def _save_quantized_df(df, saveas):
    # we can't just do df.values.tofile(saveas) because different columns can
    # have different numbers of bytes
    with open(saveas, 'wb') as f:
        for col in sorted(df.columns.values):
            f.write(df[col].values.tobytes())


def _infer_quantization_schema(in_dir, in_ext='.csv', **read_csv_kwargs):
    read_csv_kwargs.setdefault('dtype', 'object')
    col2digitstats = col_digit_stats(
        _generator_for_dfs_dir(in_dir, endswith=in_ext, **read_csv_kwargs))
    cols = sorted(list(col2digitstats.keys()))
    col2qparams = collections.OrderedDict()
    for col in cols:
        col2qparams[col] = quantization_dtype_scale(*col2digitstats[col])
    # col2qparams = {col: quantization_dtype_scale(*stats)
    #                for col, stats in col2digitstats.items()}
    return col2qparams


def load_quantized_df(path, schema, context):
    length = context['__length__']
    with open(path, 'rb') as f:
        for col in schema:
            dtype, scale = schema[col]
        # TODO pick up here; read in each col's bytes, cast to appropriate
        # dtype, and add in minval if context says there is one



# def quantize_dfs(dfs, col2digitstats):
# def quantize_dfs(dfs, col2digitstats):
def quantize_dfs(in_dir, out_dir, context_save_path, in_ext='.csv',
                 quantization_schema=None, **read_csv_kwargs):

    # col2digitstats = col_digit_stats(
    #     _generator_for_dfs_dir(in_dir, endswith=in_ext, **read_csv_kwargs))
    # col2qparams = {col: quantization_dtype_scale(*stats)
    #                for col, stats in col2digitstats.items()}

    contexts = []
    fnames = [fname for fname in os.listdir(in_dir)
              if fname.endswith(in_ext)]
    for fname in fnames:
        fid = fname.replace(in_ext, '')
        in_df_path = os.path.join(in_dir, fname)
        out_df_path = os.path.join(out_dir, fid + '.dat')
        df = pd.read_csv(in_df_path, **read_csv_kwargs)
        qdf, context = _quantize_df(df, quantization_schema)
        _save_quantized_df(qdf, out_df_path)
        context['__file_id__'] = fid
        contexts.append(context)

    df_context = pd.DataFrame.from_records(contexts)
    df_context.to_csv(context_save_path, index=False)






def main():
    pass


if __name__ == '__main__':
    main()
