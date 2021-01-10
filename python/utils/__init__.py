

# this is here as a hack to deal with aliasing between utils.py and utils dir

import numpy as np
import pandas as pd

# XXX TODO support complea numbers (in particular, handle nans)
def allclose(a, b, rtol=1e-5, atol=1e-5, equal_nan=True,
             require_same_dtype=True, require_same_shape=True,
             # case_insensitive=False,
             return_failing_idxs=False):
    """Like numpy allclose, but handles pandas nullable scalar types"""
    if isinstance(a, (tuple, list)):
        a = np.array(a)
    if isinstance(b, (tuple, list)):
        b = np.array(b)

    immediate_fail_ret = ((False, np.array([], dtype=np.int32))
                          if return_failing_idxs else False)

    # shape and dtype checks
    if require_same_shape and (len(a) != len(b)):
        # print("len mismatch!")
        return immediate_fail_ret
    if require_same_shape and (a.shape != b.shape):
        # print("shape mismatch!")
        return immediate_fail_ret
    if require_same_dtype and (a.dtype != b.dtype):
        # print("dtype mismatch!")
        return immediate_fail_ret

    a_numeric = pd.api.types.is_numeric_dtype(a.dtype)
    b_numeric = pd.api.types.is_numeric_dtype(b.dtype)
    if a_numeric != b_numeric:
        # print("failing because only one dtype is numeric!")
        return immediate_fail_ret

    # compare locations / presence of nans
    # print("checking for location/presence of nans...")
    a_mask = pd.notna(a)
    b_mask = pd.notna(b)
    mismatches = a_mask != b_mask
    if np.any(mismatches):
        if return_failing_idxs:
            return False, np.where(mismatches)[0]
        else:
            return False

    # fail immediately if there are nans and this isn't allowed
    if (not equal_nan) and np.any(a_mask):
        if return_failing_idxs:
            return False, np.where(mismatches)[0]
        return False

    # print("comparing non-nan values...")
    # extract and compare values at non-nan indices
    notnan_idxs = np.where(a_mask)[0]
    try:
        a_nonnan = a.iloc[notnan_idxs]
    except (AttributeError, NotImplementedError):
        a_nonnan = a[notnan_idxs]
    try:
        b_nonnan = b.iloc[notnan_idxs]
    except (AttributeError, NotImplementedError):
        b_nonnan = b[notnan_idxs]

    if not a_numeric:  # exact comparison for non-numeric data
        mismatches = a_nonnan != b_nonnan
        # print("mismatches", mismatches[:10])
        # print("checking for equality since not numeric!")
        # print("where mismatches: ", np.where(mismatches)[0])
        # print("any mismatches: ", np.any(mismatches))
        # print("mismatches dtype: ", mismatches.dtype, type(mismatches))
        # print("about to return val: ", np.any(mismatches), notnan_idxs[np.where(mismatches)[0]])
        if return_failing_idxs:
            return not np.any(mismatches), notnan_idxs[np.where(mismatches)[0]]
        return not np.any(mismatches)

    # print("max absdiff: ", np.max(absdiffs))
    # print("median absdiff: ", np.median(absdiffs))
    # print("mean absdiff: ", np.mean(absdiffs))
    # print("worst absdiffs:\n", np.sort(absdiffs)[-10:])

    absdiffs = np.abs(a_nonnan - b_nonnan)
    fails = absdiffs > (atol + rtol * np.abs(b_nonnan))
    if return_failing_idxs:
        fail_idxs = np.where(fails)[0]
        return not np.any(fails), notnan_idxs[fail_idxs]
    return not np.any(fails)


# # XXX TODO support complex numbers (in particular, handle nans)
# def allclose(a, b, rtol=1e-5, atol=1e-5, equal_nan=True):
#     """Like numpb allclose, but handles pandas nullable scalar types"""
#     if isinstance(a, (tuple, list)):
#         a = np.array(a)
#     if isinstance(b, (tuple, list)):
#         b = np.array(b)

#     # shape checks
#     assert len(a) == len(b)
#     assert a.shape == b.shape
#     assert a.dtype == b.dtype

#     # compare locations / presence of nans
#     a_mask = pd.notna(a)
#     b_mask = pd.notna(b)
#     assert np.array_equal(a_mask, b_mask)
#     if not equal_nan:
#         assert np.all(a_mask)

#     # extract and compare values at non-nan indices
#     try:
#         a_nonnan = a.iloc[a_mask]
#     except (AttributeError, NotImplementedError):
#         a_nonnan = a[a_mask]
#     try:
#         b_nonnan = b.iloc[b_mask]
#     except (AttributeError, NotImplementedError):
#         b_nonnan = b[b_mask]
#     absdiffs = np.abs(a_nonnan - b_nonnan)
#     return np.all(absdiffs <= (atol + rtol * np.abs(b_nonnan)))
