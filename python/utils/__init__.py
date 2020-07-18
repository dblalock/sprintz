

# this is here as a hack to deal with aliasing between utils.py and utils dir

import numpy as np
import pandas as pd


# XXX TODO support complex numbers (in particular, handle nans)
def allclose(a, b, rtol=1e-5, atol=1e-5, equal_nan=True):
    """Like numpb allclose, but handles pandas nullable scalar types"""
    if isinstance(a, (tuple, list)):
        a = np.array(a)
    if isinstance(b, (tuple, list)):
        b = np.array(b)

    # shape checks
    assert len(a) == len(b)
    assert a.shape == b.shape
    assert a.dtype == b.dtype

    # compare locations / presence of nans
    a_mask = pd.notna(a)
    b_mask = pd.notna(b)
    assert np.array_equal(a_mask, b_mask)
    if not equal_nan:
        assert np.all(a_mask)

    # extract and compare values at non-nan indices
    try:
        a_nonnan = a.iloc[a_mask]
    except (AttributeError, NotImplementedError):
        a_nonnan = a[a_mask]
    try:
        b_nonnan = b.iloc[b_mask]
    except (AttributeError, NotImplementedError):
        b_nonnan = b[b_mask]
    absdiffs = np.abs(a_nonnan - b_nonnan)
    return np.all(absdiffs <= (atol + rtol * np.abs(b_nonnan)))
