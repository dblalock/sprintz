#!/usr/bin/env python

import numpy as np
import pandas as pd


def _canonicalize(dtype):
    try:
        # put Series first so that, eg, 'Int8' properly gets mapped to the
        # nullable type rather than the numpy non-nullable 'int8' type
        s = pd.Series(data=[], dtype=dtype)
        return s.dtype
    except TypeError:
        pass
    try:
        # don't think this should handle anything a Series can't, but leave
        # this here to make sure
        a = np.empty(1, dtype=dtype)
        return a.dtype
    except TypeError:
        pass

    raise ValueError(f"Dtype '{dtype}' ({type(dtype)}) invalid for both "
                     "Numpy arrays and Pandas series.")

# _FLOAT_TYPES = [np.float16, np.float32, np.float64]

_c = _canonicalize  # shorten below definitions

_FLOAT_TYPES = [_c(np.float16), _c(np.float32), _c(np.float64)]

# _UNSIGNED_INT_TYPES = [np.uint8, np.uint16, np.uint32, np.uint64]
# _SIGNED_INT_TYPES = [np.int8, np.int16, np.int32, np.int64]
# _UNSIGNED_INT_TYPES = [np.uint8(), np.uint16(), np.uint32(), np.uint64()]
# _SIGNED_INT_TYPES = [np.int8(), np.int16(), np.int32(), np.int64()]
_UNSIGNED_INT_TYPES = [_c(np.uint8), _c(np.uint16),
                       _c(np.uint32), _c(np.uint64)]
_SIGNED_INT_TYPES = [_c(np.int8), _c(np.int16), _c(np.int32), _c(np.int64)]
_NONNULLABLE_INT_TYPES = _UNSIGNED_INT_TYPES + _SIGNED_INT_TYPES

# _NULLABLE_SIGNED_INT_TYPES = [
#     pd.Int8Dtype(), pd.Int16Dtype(), pd.Int32Dtype(), pd.Int64Dtype()]
# _NULLABLE_UNSIGNED_INT_TYPES = [
#     pd.UInt8Dtype(), pd.UInt16Dtype(), pd.UInt32Dtype(), pd.UInt64Dtype()]
# _NULLABLE_SIGNED_INT_TYPES = [_c(pd.Int8Dtype), _c(pd.Int16Dtype),
#                               _c(pd.Int32Dtype), _c(pd.Int64Dtype)]
# _NULLABLE_UNSIGNED_INT_TYPES = [_c(pd.UInt8Dtype), _c(pd.UInt16Dtype),
#                                 _c(pd.UInt32Dtype), _c(pd.UInt64Dtype)]

# have to use string dtype codes, rather than, eg, pd.Int8Dtype or
# pd.Int8Dtype() or resulting dtype is np.object; because pandas dtypes are
# insane
_NULLABLE_SIGNED_INT_TYPES = [
    _c('Int8'), _c('Int16'), _c('Int32'), _c('Int64')]
_NULLABLE_UNSIGNED_INT_TYPES = [
    _c('UInt8'), _c('UInt16'), _c('UInt32'), _c('UInt64')]
_NULLABLE_INT_TYPES = _NULLABLE_UNSIGNED_INT_TYPES + _NULLABLE_SIGNED_INT_TYPES

_NULLABLE_TO_NONNULLABLE_INT_DTYPE = dict(zip(
    _NULLABLE_INT_TYPES, _NONNULLABLE_INT_TYPES))
_NONNULLABLE_TO_NULLABLE_INT_DTYPE = dict(zip(
    _NONNULLABLE_INT_TYPES, _NULLABLE_INT_TYPES))

# _NUMERIC_DTYPES = _NONNULLABLE_INT_TYPES + _NULLABLE_INT_TYPES + _FLOAT_TYPES

# NOTE: np.bool is alias of python bool, while np.bool_ is custom a numpy type
# _BOOLEAN_DTYPES = [np.dtype(np.bool), np.dtype(np.bool_), pd.BooleanDtype()]
_BOOLEAN_DTYPES = [_c(np.bool), _c(np.bool_), _c('boolean')]

_SIGNED_EQUIVALENT = {
    _c(np.uint8):  _c(np.int8),  _c(np.int8):  _c(np.int8),  # noqa
    _c(np.uint16): _c(np.int16), _c(np.int16): _c(np.int16),
    _c(np.uint32): _c(np.int32), _c(np.int32): _c(np.int32),
    _c(np.uint64): _c(np.int64), _c(np.int64): _c(np.int64),
    _c(pd.UInt8Dtype):  _c(pd.Int8Dtype),   # noqa
    _c(pd.Int8Dtype):   _c(pd.Int8Dtype),   # noqa
    _c(pd.UInt16Dtype): _c(pd.Int16Dtype),  # noqa
    _c(pd.Int16Dtype):  _c(pd.Int16Dtype),  # noqa
    _c(pd.UInt32Dtype): _c(pd.Int32Dtype),  # noqa
    _c(pd.Int32Dtype):  _c(pd.Int32Dtype),  # noqa
    _c(pd.UInt64Dtype): _c(pd.Int64Dtype),  # noqa
    _c(pd.Int64Dtype):  _c(pd.Int64Dtype)}  # noqa
_UNSIGNED_EQUIVALENT = {
    _c(np.uint8):  _c(np.uint8),  _c(np.int8):  _c(np.uint8),  # noqa
    _c(np.uint16): _c(np.uint16), _c(np.int16): _c(np.uint16),
    _c(np.uint32): _c(np.uint32), _c(np.int32): _c(np.uint32),
    _c(np.uint64): _c(np.uint64), _c(np.int64): _c(np.uint64),
    _c(pd.UInt8Dtype):  _c(pd.UInt8Dtype),  # noqa
    _c(pd.Int8Dtype):   _c(pd.UInt8Dtype),  # noqa
    _c(pd.UInt16Dtype): _c(pd.UInt16Dtype), # noqa
    _c(pd.Int16Dtype):  _c(pd.UInt16Dtype), # noqa
    _c(pd.UInt32Dtype): _c(pd.UInt32Dtype), # noqa
    _c(pd.Int32Dtype):  _c(pd.UInt32Dtype), # noqa
    _c(pd.UInt64Dtype): _c(pd.UInt64Dtype), # noqa
    _c(pd.Int64Dtype):  _c(pd.UInt64Dtype)} # noqa

# _SIGNED_EQUIVALENT = {
#     np.uint8():  np.int8(),  np.int8():   np.int8(),  # noqa
#     np.uint16(): np.int16(), np.int16():  np.int16(),
#     np.uint32(): np.int32(), np.int32():  np.int32(),
#     np.uint64(): np.int64(), np.int64():  np.int64(),
#     pd.UInt8Dtype():  pd.Int8Dtype(),  pd.Int8Dtype():  pd.Int8Dtype(),  # noqa
#     pd.UInt16Dtype(): pd.Int16Dtype(), pd.Int16Dtype(): pd.Int16Dtype(),
#     pd.UInt32Dtype(): pd.Int32Dtype(), pd.Int32Dtype(): pd.Int32Dtype(),
#     pd.UInt64Dtype(): pd.Int64Dtype(), pd.Int64Dtype(): pd.Int64Dtype()}
# _UNSIGNED_EQUIVALENT = {
#     np.uint8():  np.uint8(),  np.int8():   np.uint8(),  # noqa
#     np.uint16(): np.uint16(), np.int16():  np.uint16(),
#     np.uint32(): np.uint32(), np.int32():  np.uint32(),
#     np.uint64(): np.uint64(), np.int64():  np.uint64(),
#     pd.UInt8Dtype():  pd.UInt8Dtype(),  pd.Int8Dtype():  pd.UInt8Dtype(),# noqa
#     pd.UInt16Dtype(): pd.UInt16Dtype(), pd.Int16Dtype(): pd.UInt16Dtype(),
#     pd.UInt32Dtype(): pd.UInt32Dtype(), pd.Int32Dtype(): pd.UInt32Dtype(),
#     pd.UInt64Dtype(): pd.UInt64Dtype(), pd.Int64Dtype(): pd.UInt64Dtype()}


# _PANDAS_NULLABLE_DTYPES_INSTANTIATED = [t() for t in _NULLABLE_INT_TYPES] +

# _SUPPORTED_PANDAS_DTYPES = _NULLABLE_INT_TYPES + [pd.BooleanDtype()]




def nullable_equivalent(dtype):
    if is_nullable(dtype):
        return dtype

    # TODO support nullable strings and other pandas dtypes
    # if dtype in _FLOAT_TYPES:
    if is_boolean(dtype):
        return _c('boolean')
    if is_float(dtype):
        return _c(dtype)

    dtype = _canonicalize(dtype)
    # if dtype in _NULLABLE_INT_TYPES:
        # return dtype
    return _NULLABLE_TO_NONNULLABLE_INT_DTYPE[dtype]


def nonnullable_equivalent(dtype):
    if not is_nullable(dtype):
        return dtype


    if is_boolean(dtype):
        return _c(np.bool_)
    if is_float(dtype):
        return _c(dtype)

    dtype = _canonicalize(dtype)
    # if dtype in _NONNULLABLE_INT_TYPES:
        # return dtype


    # print("dtype: ", dtype, type(dtype))
    # print("nullable int dtypes: ")
    # for t in _NULLABLE_INT_TYPES:
    #     print(t, type(t))
    # print("dtype in nullable ints: ", dtype in _NULLABLE_INT_TYPES)


    return _NULLABLE_TO_NONNULLABLE_INT_DTYPE[dtype]


def signed_equivalent(dtype):
    dtype = _canonicalize(dtype)
    return _SIGNED_EQUIVALENT[dtype]


def unsigned_equivalent(dtype):
    dtype = _canonicalize(dtype)
    return _UNSIGNED_EQUIVALENT[dtype]


def is_complex(dtype):
    return pd.api.types.is_complex_dtype(dtype)


def is_float(dtype):
    return pd.api.types.is_float_dtype(dtype)
    # return _canonicalize(dtype) in _FLOAT_TYPES


def is_numeric(dtype):
    # print("is_numeric: checking dtype: ", dtype)

    if is_boolean(dtype):
        return False

    # dtype = _canonicalize(dtype)
    # if _canonicalize(dtype) in _BOOLEAN_DTYPES:
    # _ = _BOOLEAN_DTYPES
    # for btype in _BOOLEAN_DTYPES:
    #     print("dtype, btype: ", dtype, btype)
    #     if btype == dtype:
    #         return False

    # if dtype in _BOOLEAN_DTYPES:
    #     # exclude bools since they mess up quantization and just generally
    #     # don't act like numbers
    #     return False
    return pd.api.types.is_numeric_dtype(dtype)


def is_boolean(dtype):
    # TODO this says Categorical([True, False]) is boolean; do we want that?
    return pd.api.types.is_bool_dtype(dtype)


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

    # if _canonicalize(dtype) in _FLOAT_TYPES:
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
    # if dtype in _FLOAT_TYPES:
    #     return True

    # # XXX include other nullable dtypes

    # return False


# used for codec type whitelists/blacklists
# note that typelist can contain types, unary functions of types, and
# keywords like "numeric"
# note that this is an OR of whether it matches each one, not AND
def dtype_in_list(dtype, typelist):
    dtype = _canonicalize(dtype)
    # typelist = [_canonicalize(dtype_or_func) for dtype_or_func in typelist]

    # if dtype in typelist:
    #     return True  # easy case; dtype is in typelist

    # print('dtype:', dtype)
    # print('typelist:\n', typelist)

    for type_or_func in typelist:
        if callable(type_or_func):
            # print(f"'{type_or_func}' is callable!")
            f = type_or_func
            if f(dtype):
                return True
            continue

        # print(f"'{type_or_func}' isn't callable!")
        typ = type_or_func
        # print("dtype_in_list: got type: ", typ)
        keyword2func = {
            'numeric': is_numeric,
            'anyint': is_int,
            'signedint': is_signed_int,
            'unsignedint': is_unsigned_int,
            'complex': is_complex,
            'anyfloat': is_float,
            'nullable': is_nullable,
            'nonnullable': lambda x: not is_nullable(x),
            np.object: is_object}

        for k in keyword2func:
            if typ == k and keyword2func[k](dtype):
                return True
        if typ in keyword2func:
            continue  # was in dict, but predicate was false

        # print("typ: ", typ)
        # print("typ in dict: ", typ in keyword2func)

        # if typ == 'numeric':
        #     if is_numeric(dtype):
        #         return True
        #     else:
        #         continue
        # if typ == 'anyint':
        #         is_int(dtype):
        #     return True
        # if typ == 'signedint' and is_signed_int(dtype):
        #     return True
        # if typ == 'unsignedint' and is_unsigned_int(dtype):
        #     return True
        # if typ == 'complex' and is_complex(dtype):
        #     return True
        # if typ == 'anyfloat' and is_float(dtype):
        #     return True
        # if typ == 'anybool' and is_boolean(dtype):
        #     return True
        # if typ == 'nullable' and is_nullable(dtype):
        #     return True
        # if typ == 'nonnullable' and not is_nullable(dtype):
        #     return True
        # if typ == np.object and is_object(dtype):
        #     return True

        # rhs wasn't a func or a special keyword; assume regular old type
        if dtype == _canonicalize(type_or_func):
            return True

    return False



        # f = to_func()


