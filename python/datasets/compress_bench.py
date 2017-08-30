#!/usr/bin/env python

from __future__ import division

import os
import numpy as np
from joblib import Memory

from ..utils.files import ensure_dir_exists

from . import ampds, ucr, pamap, uci_gas, msrc
from . import paths

_memory = Memory('.', verbose=1)

# SAVE_DIR = paths.COMPRESSION_ROWMAJOR_DIR

# This file writes out the datasets used in the experiments, including:k
#
# UCR datasets (ucr/)
#   -all examples concatenated (with interp)
#   -one file per data
# MSRC (msrc/), PAMAP (pamap/)
#   -all recordings concatenated (with interp), excluding timestamps and annotations
# AMPDS
#   -{weather, electricity, gas, water}, excluding timestamps
# UCI Gas
#   -both gas levels and sensor readings (but not the timestamps)
# ECG
#   -TODO
#
# All files are just dumps of the raw binary (numpy tofile()) in
# little-endian byte order. All files are available as:
#   {row-major, column-major} x {8bit, 16bit}
# where row-major means that all variables at a given timestamp or stored
# contiguously and column-major means that all samples from a given variable
# are stored contiguously.
#
# Note that the UCR datasets are only in "colmajor" since they're univariate,
# so row-major and column-major are the same.

def _quantize(mat, dtype):
    mat -= np.min(mat, axis=0)
    mat = mat.astype(np.float32)
    mat /= np.max(mat, axis=0)
    if dtype == np.uint8:
        max_val = 255
    elif dtype == np.uint16:
        max_val = 65535
    else:
        raise ValueError("Invalid dtype '{}'".format(dtype))

    return (mat * max_val).astype(dtype)


def _ensure_list_or_tuple(x):
    if not isinstance(x, (list, tuple)):
        return [x]
    return x


def write_datasets(mats_list, dset_names, dtype=np.uint16, order='c', subdir=''):
    mats_list = _ensure_list_or_tuple(mats_list)
    dset_names = _ensure_list_or_tuple(dset_names)

    dtype_names = {np.uint8: 'uint8', np.uint16: 'uint16'}

    for name, mat in zip(dset_names, mats_list):
        mat = _quantize(mat, dtype=dtype)

        if order == 'c':
            savedir = paths.COMPRESSION_ROWMAJOR_DIR
        elif order == 'f':
            savedir = paths.COMPRESSION_COLMAJOR_DIR
            mat = np.asfortranarray(mat)
        else:
            raise ValueError("Unrecognized order '{}'".format(order))

        savedir = os.path.join(savedir, dtype_names[dtype])
        if subdir:
            savedir = os.path.join(savedir, subdir)
        ensure_dir_exists(savedir)
        path = os.path.join(savedir, name + '.dat')
        mat.tofile(path)


# def ucr_dataset_to_dataset(X, interp_npoints=5):
#     # difference between end of each example and start of next one
#     boundary_jumps = X[1:, 0] - X[:-1, -1]

#     # if interp_npoints == 'infer':
#     #     diffs = np.diff(X, axis=-1)
#     #     max_diff = np.max(np.abs(diffs))
#     #     max_boundary_jump = np.max(np.abs(boundary_jumps))
#     #     interp_npoints = int(np.ceil(max_boundary_jump / max_diff))
#     # interp_npoints = max(1, interp_npoints)  # shouldn't be necessary

#     offsets = np.arange(1., interp_npoints + 1.) / (interp_npoints + 1)

#     # get samples as outer product of size of the gaps to interpolate for
#     # each row and the interpolated coefficients, which are between 0 and 1
#     lhs = np.append(boundary_jumps, 0).reshape(-1, 1)  # col vect
#     rhs = offsets.reshape(1, -1)  # row vect
#     interp_samples = np.dot(lhs, rhs)

#     return np.hstack((X, interp_samples)).ravel()

@_memory.cache
def concat_and_interpolate(mats, interp_npoints=5):
    # assumes each row of each mat is one time step and mats is a list
    dtype = mats[0].dtype

    first_vals = np.vstack([mat[0] for mat in mats])
    last_vals = np.vstack([mat[-1] for mat in mats])
    boundary_jumps = first_vals[1:] - last_vals[:-1]

    # print "first_vals: ", first_vals
    # print "last_vals: ", last_vals
    # print "boundary_jumps: ", boundary_jumps

    offsets = np.arange(1., interp_npoints + 1.) / (interp_npoints + 1)

    # print "offsets: ", boundary_jumps

    # multiply jumps by offsets to get interpolated values; note that
    # we reshape offsets oddly to get it to broadcast
    new_shape = list(np.ones(len(boundary_jumps.shape), dtype=np.int))
    new_shape.append(len(offsets))
    offsets = offsets.reshape(new_shape)
    boundary_jumps = boundary_jumps[..., np.newaxis]
    interp_samples = (offsets * boundary_jumps).astype(dtype)
    interp_samples += last_vals[:-1][..., np.newaxis]

    if len(mats[0].shape) < 2:
        mats = [mat[..., np.newaxis] for mat in mats]

    out_mats = [mats[0]]
    for i in range(1, len(mats)):
        if i == 1:
            print "interpolated samples shape: ", interp_samples[i - 1].T.shape
            print "data matrix shape: ", mats[i].shape
        out_mats.append(interp_samples[i - 1].T)
        out_mats.append(mats[i])

    return np.vstack(out_mats)



# ================================================================ main

def _test_concat_and_interpolate():  # TODO less janky unit tests
    X = np.arange(12).reshape(4, 3)
    mats = [X, X + 9 + 6]
    ret = concat_and_interpolate(mats)
    assert len(ret) == 2 * len(X) + 5
    assert np.array_equal(X, ret[:4])
    assert np.array_equal(X + 15, ret[-4:])

    X = np.arange(0, 31, 6).reshape(2, 3)
    ret = concat_and_interpolate(X)
    ans = np.array([0, 6, 12, 13, 14, 15, 16, 17, 18, 24, 30])[..., np.newaxis]
    assert np.array_equal(ret, ans)

def mat_from_recordings(recs):
    return concat_and_interpolate([r.data for r in recs])


def main():
    _test_concat_and_interpolate()

    # gas_recs = ampds.all_gas_recordings()
    # recs = ampds.all_weather_recordings()
    # print "recs:", [r.data.shape for r in gas_recs]
    # mat = concat_and_interpolate([r.data[:, 1:] for r in recs])

    # mat = mat_from_recordings(ampds.all_gas_recordings())[:, 1:]
    # mat = mat_from_recordings(ampds.all_gas_recordings())[:, 1:]

    funcs_and_names = [
        # (ampds.all_gas_recordings, 'ampd_gas'),
        # (ampds.all_water_recordings, 'ampd_water'), # TODO get it working
        # (ampds.all_power_recordings, 'ampd_power'),
        # (ampds.all_weather_recordings, 'ampd_weather'),
        # (uci_gas.all_recordings, 'uci_gas'),
        # (pamap.all_recordings, 'pamap'),
        (msrc.all_recordings, 'msrc'),
    ]

    STORAGE_ORDER = 'f'

    for func, name in funcs_and_names:
        recordings = func()
        mat = mat_from_recordings(recordings)
        for dtype in (np.uint8, np.uint16):
            write_datasets(mat, name, dtype=dtype, order=STORAGE_ORDER)

    # for dset in ucr.allUCRDatasets(): # TODO uncomment
    for dset in ucr.origUCRDatasets():
        mat = concat_and_interpolate(dset.X)
        for dtype in (np.uint8, np.uint16):
            write_datasets(mat, dset.name, dtype=dtype, order=STORAGE_ORDER, subdir='ucr')


    # import matplotlib.pyplot as plt
    # plt.plot(mat, lw=.5)
    # plt.show()

if __name__ == '__main__':
    main()
