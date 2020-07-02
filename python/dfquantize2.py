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
                  lossless_tol=1e-4):
    orig_dtype = x.dtype
    offset = x.min() if offset is None else offset
    x_offset = x - offset

    # assert method in ('lossless_base10', 'rescale_u8', 'rescale_u16')

    if not isinstance(scale, str):
        pass  # scale to use is supplied directly
    elif scale == 'rescale_u8':
        scale = 255. / x_offset.max()
        dtype = np.uint8
    elif scale == 'rescale_u16':
        scale = 65525. / x_offset.max()
        dtype = np.uint16
    elif scale == 'lossless_base10':
        # this is the tricky one; need to infer how many base10 decimal places
        # were recorded
        # for shift in range(64):
        for shift in range(20):
            scale = 10. ** shift
            x_scaled = x_offset * scale
            x_scaled_ints = np.round(x_scaled).astype(np.int64)
            diffs = x_scaled - x_scaled_ints
            if np.abs(diffs).max() < lossless_tol:
                break
    else:
        raise ValueError(
            f"Scale must be a number or valid string; got '{scale}'")

    if dtype is None:
        maxval = (x_offset * scale).max()
        if maxval <= 255:
            dtype = np.uint8
        if maxval <= 65525:
            dtype = np.uint16
        elif maxval <= ((1 << 32) - 1):
            dtype = np.uint32
        elif maxval <= ((1 << 64) - 1):
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


def quantize(x, qparams):
    x = x - qparams.offset
    x = x * qparams.scale  # not in place so that float scale works on int x
    return np.round(x).astype(qparams.dtype)


def unquantize(x, qparams):
    x = x.astype(np.float64)
    x *= (1. / qparams.scale)
    x = x.astype(qparams.orig_dtype)
    x += qparams.offset
    return x
    # return x.astype(qparams.orig_dtype)


def mask_for_dtype(dtype):
    if not isinstance(dtype, np.dtype):
        dtype = dtype.type  # handle dtype object vs its type
    return {np.uint8: 0xff, np.uint16: 0xffff,
            np.uint32: 0xffffffff}.get(dtype)
    # d = {np.dtype(k): v for k, v in d.items()}
    # return d.get(dtype)

# def test_quantize_unquantize
