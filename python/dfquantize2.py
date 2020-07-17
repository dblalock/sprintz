#!/usr/bin/env python

import collections
import os
import numpy as np
import numba
import pandas as pd

from python import dtypes


class QuantizeParams(collections.namedtuple(
        'QuantizeParams', 'dtype offset scale orig_dtype allfinite'.split())):
    pass


def infer_qparams(x, offset=None, scale='lossless_base10', dtype=None,
                  allow_nan_inf=True):
    orig_dtype = x.dtype

    # print("x dtype, is_boolean, is_nullable", x.dtype, dtypes.is_boolean(x.dtype), dtypes.is_nullable(x.dtype))
    if dtypes.is_boolean(x.dtype):
        # allfinite = dtypes.is_nullable(x.dtype)
        allfinite = (not np.any(pd.isna(x))) and (not np.any(np.isinf(x)))
        return QuantizeParams(dtype=np.uint8, offset=0, scale=1,
                              orig_dtype=orig_dtype, allfinite=allfinite)

    u8_max = 255
    u16_max = 65535
    u32_max = (1 << 32) - 1
    u64_max = (1 << 63) + ((1 << 63) - 1)  # not a float since too large
    if not allow_nan_inf:
        # assert not np.any(np.isnan(x))
        assert not np.any(np.isinf(x))
        assert not np.any(pd.isna(x))
        allfinite = True
    else:
        # mask = np.isfinite(x)
        # mask = pd.notna(x)
        # mask = np.isfinite(x)
        # if not np.all(mask):
            # dtype could also be object, but in that case we'll fail
            # at quantizing it anyway, so might as well fail fast here
            # print("x dtype: ", x.dtype)
            # assert x.dtype.type in (np.float32, np.float64)  # nans in int ar?

        mask = pd.notna(x)
        x = x[mask]
        if len(x) < 1:
            # all values are nan or inf; might as well encode them
            # with just one byte each
            # XXX this conflates nan, inf, and -inf
            return QuantizeParams(
                dtype=np.uint8, offset=np.nan, allfinite=False,
                scale=np.nan, orig_dtype=orig_dtype)
            # raise ValueError("input contained no finite values. "
            #                  "Cannot infer parameters")
        allfinite = len(x) == len(mask)
        if not allfinite: # at least one nan or inf
            u8_max -= 1
            u16_max -= 1
            u32_max -= 1
            u64_max -= 1

    # print("qparams: type(x), x.dtype", type(x), x.dtype)
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
            # less than 1% unexplained
            if np.log10(np.abs(diffs).max() + 1e-20) < -2:
                break
        scale = int(scale)
    else:
        raise ValueError(
            f"Scale must be a number or valid string; got '{scale}'")

    # print("x maxval, shift, scale: ", x.max(), shift, scale)
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
                             "too large to quantize (originally "
                             f"~{maxval / scale + offset}")

    # return QuantizeParams(dtype=dtype, offset=offset, scale=scale,
    #                       orig_dtype=orig_dtype)

    ret = QuantizeParams(dtype=dtype, offset=offset, scale=scale,
                         orig_dtype=orig_dtype, allfinite=allfinite)
    # print("inferred qparams: ", ret)
    return ret


def _naninf_val_for_dtype(dtype):
    if isinstance(dtype, np.dtype):
        # print("got dtype: ", dtype, type(dtype))
        dtype = dtype.type  # handle dtype object vs its type
    return {np.uint8: 255,
            np.uint16: 65535,
            np.uint32: ((1 << 32) - 1),
            np.uint64: (1 << 63) + ((1 << 63) - 1),
            np.float16: np.nan,
            np.float32: np.nan,
            np.float64: np.nan,
            }.get(dtype)


def quantize(x, qparams):
    x = x - qparams.offset
    x = x * qparams.scale  # not in place so that float scale works on int x
    mask = pd.notna(x)
    ret = np.empty(x.shape, dtype=qparams.dtype)
    # if pd.api.types.is_integer_dtype(qparams.dtype):
    #     # print("mask: ", mask, type(mask), mask.dtype, pd.isna(mask).sum())
    #     # print("x[mask]: ", x[mask], type(x[mask]), x[mask].dtype, pd.isna(x[mask]).sum())
    #     if dtypes.is_float(x.dtype):
    #         x = np.round(x[mask])
        # ret[mask] = x[mask]
        # print("x[mask]: ", x[mask])
    # else:
        # ret[mask] = x[mask]
    # x = np.round(x).astype(qparams.dtype)

    if dtypes.is_int(qparams.dtype) and dtypes.is_float(x.dtype):
        ret[mask] = np.round(x[mask])
    else:
        ret[mask] = x[mask]
    ret[~mask] = _naninf_val_for_dtype(qparams.dtype)
    # print("quantize: using nanval: ", _naninf_val_for_dtype(qparams.dtype))
    return ret


def _is_power_of_2(x):
    try:
        n = int(x)
        if n != x:
            return False
        return (n & (n - 1) == 0) and (n != 0)
    except ValueError:
        return False


def unquantize(x, qparams):
    # unpack series into np array; series astype() is inadquate, and
    # if input has been quantized, guaranteed to be a numpy type (not a
    # nullable int) so no info loss
    try:
        x = x.values
    except AttributeError:
        pass

    print("got x dtype, qparams", x.dtype, qparams)
    assert x.dtype == qparams.dtype
    if dtypes.is_int(qparams.orig_dtype):
        assert qparams.scale <= 1  # no reason to expand ints
    # if dtypes.is_float(qparams.scale):
    #     ret = x.astype(np.float64)
    #     ret *= (1. / qparams.scale)
    # else:
    #     ret

    # if dtypes.is_int(qparams.orig_dtype) and dtypes.is_int(x.dtype):


    # if dtypes.scale(qparams.scale) != 1:
    #     ret = x.astype(np.float64)
    #     ret *= (1. / qparams.scale)

    # three vars to think about:
    #   -quantized dtype is int
    #   -orig dtype is int
    #   -scale is 1
    #   -scale is power of 2
    #
    # observe that, for original type of int, scale <= 1

    # if not qparams.allfinite and dtypes.is_nullable(qparams.orig_dtype):
    if qparams.allfinite:
        # print("no naninfs!")
        naninf_mask = np.zeros(len(x)) != 0  # all False
        x_nan = np.array([], dtype=x.dtype)
        x_notnan = x
    else:
        naninf_mask = x == _naninf_val_for_dtype(qparams.dtype)
        # print("computed naninf_mask: ", naninf_mask)
        x_nan = x[naninf_mask]
        x_notnan = x[~naninf_mask]

    if qparams.scale != 1:
        x_notnan = x_notnan.astype(np.float64)
        x_notnan *= (1. / qparams.scale)

    # print("unquantize: qparams: ", qparams)
    # print("unquantize: orig dtype nullable: ", dtypes.is_nullable(qparams.orig_dtype))
    # print("unquantize: naninf val:   ", _naninf_val_for_dtype(qparams.dtype))
    # print("unquantize: naninf mask:  ", naninf_mask)
    # print("unquantize: ~naninf mask: ", ~naninf_mask)
    # print("unquantize: x before unquantize: ", x, x.dtype)
    # print("unquantize: x type, dtype", type(x), x.dtype)
    # print("unquantize: scale power of two?: ", _is_power_of_2(qparams.scale))

    # pandas series astype() lacks unsafe casting, so we have to create
    # numpy array first, and then insert it into the series we return
    cast_dtype = dtypes.nonnullable_equivalent(qparams.orig_dtype)
    x_notnan = x_notnan.astype(cast_dtype, casting='unsafe')
    x_notnan += qparams.offset

    ret = pd.Series(np.empty(len(x), dtype=cast_dtype), dtype=qparams.orig_dtype)
    # print("unquantize: initial ret dtype: ", ret.dtype)

    # # print("naninf_mask: ", naninf_mask)
    # print("x notnan: ", x_notnan)
    # print("ret notnan: ", ret[~naninf_mask])
    # # print("x notnan: ", x_notnan)

    ret.iloc[~naninf_mask] = x_notnan
    # print("ret before naninf assign: ", ret)

    # EDIT: even when naninf mask is all zeros, this turns int-valued ret
    # into floats
    if not qparams.allfinite:
        # works even for pd nullable ints; pd.NA fails on floats
        ret.iloc[naninf_mask] = np.nan

    # print("ret: ", ret)

    assert ret.dtype == qparams.orig_dtype

    # print("unquantize: cast_dtype: ", cast_dtype)
    # print("unquantize: ret dtype: ", ret.dtype)

    return ret

    # # print("unquantize: using nanval: ", _naninf_val_for_dtype(qparams.dtype))

    # ret = pd.Series(ret)
    # if dtypes.is_nullable(qparams.orig_dtype) and not qparams.allfinite:
    #     # print("yep, need to replace nans!")
    #     # no nan values for ints
    #     naninf_mask = x == _naninf_val_for_dtype(qparams.dtype)
    #     # print("naninf_mask: ", naninf_mask)
    #     ret[naninf_mask] = np.nan
    # return ret
    # return x.astype(qparams.orig_dtype)


# def mask_for_dtype(dtype):
#     if not isinstance(dtype, np.dtype):
#         dtype = dtype.type  # handle dtype object vs its type
#     return {np.uint8: 0xff, np.uint16: 0xffff,
#             np.uint32: 0xffffffff}.get(dtype)
#     # d = {np.dtype(k): v for k, v in d.items()}
#     # return d.get(dtype)

# # def test_quantize_unquantize
