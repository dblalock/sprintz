#!/usr/bin/env python

import numpy as np
import pandas as pd

FLOAT_TYPES = [np.float16, np.float32, np.float64]

_UNSIGNED_INT_TYPES = [np.uint8, np.uint16, np.uint32, np.uint64]
_SIGNED_INT_TYPES = [np.int8, np.int16, np.int32, np.int64]
_NONNULLABLE_INT_TYPES = _UNSIGNED_INT_TYPES + _SIGNED_INT_TYPES

_NULLABLE_SIGNED_INT_TYPES = [
    pd.Int8Dtype, pd.Int16Dtype, pd.Int32Dtype, pd.Int64Dtype]
_NULLABLE_UNSIGNED_INT_TYPES = [
    pd.UInt8Dtype, pd.UInt16Dtype, pd.UInt32Dtype, pd.UInt64Dtype]
_NULLABLE_INT_TYPES = _NULLABLE_UNSIGNED_INT_TYPES + _NULLABLE_SIGNED_INT_TYPES

_NULLABLE_TO_NONNULLABLE_INT_DTYPE = dict(zip(
    _NULLABLE_INT_TYPES, _NONNULLABLE_INT_TYPES))
_NONNULLABLE_TO_NULLABLE_INT_DTYPE = dict(zip(
    _NONNULLABLE_INT_TYPES, _NULLABLE_INT_TYPES))

_NUMERIC_DTYPES = _NONNULLABLE_INT_TYPES + _NULLABLE_INT_TYPES + FLOAT_TYPES

# NOTE: np.bool is alias of python bool, while np.bool_ is custom a numpy type
_BOOLEAN_DTYPES = [np.bool, np.bool_, pd.BooleanDtype]

# _PANDAS_NULLABLE_DTYPES_INSTANTIATED = [t() for t in _NULLABLE_INT_TYPES] +

def _canonicalize(dtype):
    try:
        return dtype.type
    except AttributeError:
        return dtype


def nullable_equivalent(dtype):
    # TODO support nullable strings and other pandas dtypes
    dtype = _canonicalize(dtype)
    if dtype in FLOAT_TYPES:
        return dtype
    if dtype in _NULLABLE_INT_TYPES:
        return dtype
    return _NULLABLE_TO_NONNULLABLE_INT_DTYPE[dtype]


def nonnullable_equivalent(dtype):
    dtype = _canonicalize(dtype)
    if dtype in FLOAT_TYPES:
        return dtype
    if dtype in _NONNULLABLE_INT_TYPES:
        return dtype
    return _NULLABLE_TO_NONNULLABLE_INT_DTYPE[dtype]


def signed_equivalent(dtype):
    dtype = _canonicalize(dtype)
    return {np.uint8:  np.int8,  np.int8:   np.int8,  # noqa
            np.uint16: np.int16, np.int16:  np.int16,
            np.uint32: np.int32, np.int32:  np.int32,
            np.uint64: np.int64, np.int64:  np.int64,
            pd.UInt8Dtype:  pd.Int8Dtype,  pd.Int8Dtype:  pd.Int8Dtype,  # noqa
            pd.UInt16Dtype: pd.Int16Dtype, pd.Int16Dtype: pd.Int16Dtype,
            pd.UInt32Dtype: pd.Int32Dtype, pd.Int32Dtype: pd.Int32Dtype,
            pd.UInt64Dtype: pd.Int64Dtype, pd.Int64Dtype: pd.Int64Dtype
           }[dtype]


def unsigned_equivalent(dtype):
    dtype = _canonicalize(dtype)
    return {np.uint8:  np.uint8,  np.int8:   np.uint8,  # noqa
            np.uint16: np.uint16, np.int16:  np.uint16,
            np.uint32: np.uint32, np.int32:  np.uint32,
            np.uint64: np.uint64, np.int64:  np.uint64,
            pd.UInt8Dtype:  pd.UInt8Dtype,  pd.Int8Dtype:  pd.UInt8Dtype, # noqa
            pd.UInt16Dtype: pd.UInt16Dtype, pd.Int16Dtype: pd.UInt16Dtype,
            pd.UInt32Dtype: pd.UInt32Dtype, pd.Int32Dtype: pd.UInt32Dtype,
            pd.UInt64Dtype: pd.UInt64Dtype, pd.Int64Dtype: pd.UInt64Dtype
           }[dtype]


def is_complex(dtype):
    return pd.api.types.is_complex_dtype(dtype)


def is_float(dtype):
    return pd.api.types.is_float_dtype(dtype)
    # return _canonicalize(dtype) in FLOAT_TYPES


def is_numeric(dtype):
    return pd.api.types.is_numeric_dtype(dtype)


def is_boolean(dtype):
    return _canonicalize(dtype) in _BOOLEAN_DTYPES


def is_int(dtype):
    return pd.api.types.is_integer_dtype(dtype)


def is_signed_int(dtype):
    return pd.api.types.is_signed_integer_dtype(dtype)


def is_unsigned_int(dtype):
    return pd.api.types.is_unsigned_integer_dtype(dtype)


def is_object(dtype):
    return pd.api.types.is_object_dtype(dtype)


def is_pandas_extension_type(dtype):
    return api.types.is_extension_array_dtype(dtype)


def is_fixed_size(dtype):
    if is_object(dtype):
        return False
    return True

    # # try:
    # #     ar = np.array([], dtype=dtype)
    # #     _ = ar.itemsize
    # # # dtype = _canonicalize(dtype)

    # if is_float(dtype):
    #     return True
    # if is_complex(dtype):
    #     return True


    # # XXX string and byte dtypes can be fixed size
    # return not np.is_nullable(dtype)


def is_nullable(dtype):
    # dtype = _canonicalize(dtype)
    # if dtype in _NULLABLE_INT_TYPES:
    #     return True
    # if dtype in FLOAT_TYPES:
    #     return True

    # you'd think there would be a nice way to check whether a dtype is
    # in a list of known nullable dtypes, but for pandas nullable dtypes,
    # there basically just isn't; you get madness like:
    #   s = pd.Series(data=[1, 0, pd.NA], dtype=pd.Int8Dtype)
    # throwing a ValueError for this being an invalid dtype, while
    #   s = pd.Series(data=[1, 0, pd.NA], dtype='Int8') is fine
    # and then
    #   print(s.dtype in [pd.Int8Dtype])  # false
    #   print(s.dtype.type in [pd.Int8Dtype])  # false
    #   print(s.dtype, s.dtype.type)  # Int8 <class 'numpy.int8'>
    # also
    #   print(pd.Int8Dtype.type)  # <class 'numpy.int8'>
    # The only thing that works is checking whether the dtype is in
    # [pd.Int8Dtype()] (instantiated), but this doesn't work if we get
    # passed in a the class pd.Int8Dtype; I suppose you could try both
    # a master list of the nullable dtype classes and their instantiations,
    # but at this point I only trust actually trying to put nans in it

    # if _canonicalize(dtype) in FLOAT_TYPES:
    #     return True  # below tests fail for float dtypes

    try:
        # NOTE: pd.NA will make this throw for floats, but np.nan always works
        s = pd.Series(data=[np.nan], dtype=dtype)
        return True
    except (TypeError, ValueError):
        pass

    try:
        # try instatiating the dtype (eg, pd.Int8Dtype -> pd.Int8Dtype() )
        # NOTE: pd.NA will make this throw for floats, but np.nan always works
        s = pd.Series(data=[np.nan], dtype=dtype())
        return True
    except (TypeError, ValueError):
        pass

    return False
    # dtype = _canonicalize(dtype)
    # if dtype in _NULLABLE_INT_TYPES:
    #     return True
    # if dtype in FLOAT_TYPES:
    #     return True

    # # XXX include other nullable dtypes

    # return False


# used for codec type whitelists/blacklists
# note that typelist can contain types, unary functions of types, and
# keywords like "numeric"
# note that this is an OR of whether it matches each one, not AND
def dtype_in_list(dtype, typelist):
    dtype = _canonicalize(dtype)
    typelist = [_canonicalize(dtype_or_func) for dtype_or_func in typelist]

    if dtype in typelist:
        return True  # easy case; dtype is in typelist

    for type_or_func in typelist:
        if callable(type_or_func):
            f = type_or_func
            if f(dtype):
                return True
        else:  # not callable
            typ = type_or_func
            # print("dtype_in_list: got type: ", typ)
            if typ == 'numeric' and is_numeric(dtype):
                return True
            if typ == 'anyint' and is_int(dtype):
                return True
            if typ == 'signedint' and is_signed_int(dtype):
                return True
            if typ == 'unsignedint' and is_unsigned_int(dtype):
                return True
            if typ == 'complex' and is_complex(dtype):
                return True
            if typ == 'anyfloat' and is_float(dtype):
                return True
            if typ == 'anybool' and is_boolean(dtype):
                return True
            if typ == 'nullable' and is_nullable(dtype):
                return True
            if typ == 'nonnullable' and not is_nullable(dtype):
                return True
            if typ == np.object and is_object(dtype):
                return True

    return False



        # f = to_func()


