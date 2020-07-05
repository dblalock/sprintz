#!/usr/bin/env python

import collections
import os
import numpy as np
import numba
import pandas as pd


UNSIGNED_INT_TYPES = [np.uint8, np.uint16, np.uint32, np.uint64]
SIGNED_INT_TYPES = [np.int8, np.int16, np.int32, np.int64]
INT_TYPES = UNSIGNED_INT_TYPES + SIGNED_INT_TYPES


class QuantizeParams(collections.namedtuple(
        'QuantizeParams', 'dtype offset scale orig_dtype'.split())):
    pass


def infer_qparams(x, offset=None, scale='lossless_base10', dtype=None,
                  allow_nan_inf=True):
    orig_dtype = x.dtype

    u8_max = 255
    u16_max = 65535
    u32_max = (1 << 32) - 1
    u64_max = (1 << 63) + ((1 << 63) - 1)  # not a float since too large
    if not allow_nan_inf:
        assert not np.any(np.isnan(x))
        assert not np.any(np.isinf(x))
    else:
        mask = np.isfinite(x)
        if not np.all(mask):
            # dtype could also be object, but in that case we'll fail
            # at quantizing it anyway, so might as well fail fast here
            # print("x dtype: ", x.dtype)
            assert x.dtype.type in (np.float32, np.float64)  # nans in int ar?
        x = x[mask]
        if len(x) < 1:
            # all values are nan or inf; might as well encode them
            # with just one byte each
            return QuantizeParams(dtype=np.uint8, offset=np.nan,
                                  scale=np.nan, orig_dtype=orig_dtype)
            # raise ValueError("input contained no finite values. "
            #                  "Cannot infer parameters")
        u8_max -= 1
        u16_max -= 1
        u32_max -= 1
        u64_max -= 1

    offset = x.min() if offset is None else offset
    x_offset = x - offset

    if not isinstance(scale, str):
        pass  # scale to use is supplied directly
    elif scale == 'rescale_u8':
        scale = min(1, float(u8_max) / x_offset.max())
        dtype = np.uint8
    elif scale == 'rescale_u16':
        scale = min(1, float(u16_max) / x_offset.max())
        dtype = np.uint16
    elif scale == 'lossless_base10':
        # this is the tricky one; need to infer how many base10 decimal places
        # were recorded
        # for shift in range(64):
        # print("x_offset in lossless_base10:\n", x_offset)
        for shift in range(20):
            scale = 10. ** shift
            x_scaled = x_offset * scale
            x_scaled_ints = np.round(x_scaled).astype(np.int64)
            diffs = x_scaled - x_scaled_ints
            if np.abs(diffs).max() < 10 ** (-shift - 2):
                break
        scale = int(scale)
    else:
        raise ValueError(
            f"Scale must be a number or valid string; got '{scale}'")

    if dtype is None:
        maxval = (x_offset * scale).max()
        if maxval <= u8_max:
            dtype = np.uint8
        elif maxval <= u16_max:
            dtype = np.uint16
        elif maxval <= u32_max:
            dtype = np.uint32
        elif maxval <= u64_max:
            dtype = np.uint64
        else:
            raise ValueError(f"offset and scaled maximum value {maxval} is "
                             "large to quantize")

    # return QuantizeParams(dtype=dtype, offset=offset, scale=scale,
    #                       orig_dtype=orig_dtype)

    ret = QuantizeParams(dtype=dtype, offset=offset, scale=scale,
                         orig_dtype=orig_dtype)
    # print("inferred qparams: ", ret)
    return ret


def _naninf_val_for_dtype(dtype):
    if isinstance(dtype, np.dtype):
        # print("got dtype: ", dtype, type(dtype))
        dtype = dtype.type  # handle dtype object vs its type
    return {np.uint8: 255, np.uint16: 65535,
            np.uint32: ((1 << 32) - 1),
            np.uint64: (1 << 63) + ((1 << 63) - 1),
            }.get(dtype)


def quantize(x, qparams):
    x = x - qparams.offset
    x = x * qparams.scale  # not in place so that float scale works on int x
    mask = ~np.isfinite(x)
    x = np.round(x).astype(qparams.dtype)
    x[mask] = _naninf_val_for_dtype(qparams.dtype)
    return x


def unquantize(x, qparams):
    ret = x.astype(np.float64)
    ret *= (1. / qparams.scale)
    ret = ret.astype(qparams.orig_dtype)
    ret += qparams.offset
    if ret.dtype.type in (np.float32, np.float64):
        # no nan values for ints
        naninf_mask = x == _naninf_val_for_dtype(qparams.dtype)
        # print("naninf_mask: ", naninf_mask)
        ret[naninf_mask] = np.nan
    return ret
    # return x.astype(qparams.orig_dtype)


def mask_for_dtype(dtype):
    if not isinstance(dtype, np.dtype):
        dtype = dtype.type  # handle dtype object vs its type
    return {np.uint8: 0xff, np.uint16: 0xffff,
            np.uint32: 0xffffffff}.get(dtype)
    # d = {np.dtype(k): v for k, v in d.items()}
    # return d.get(dtype)

# def test_quantize_unquantize
