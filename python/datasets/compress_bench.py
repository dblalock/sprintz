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

def _quantize(mat, dtype, axis=0):
    # print "quantize: got mat with shape: ", mat.shape
    mat -= np.min(mat, axis=axis, keepdims=True)
    mat = mat.astype(np.float32)
    mat /= np.max(mat, axis=axis, keepdims=True)
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


def write_dataset(mat, name, dtypes=(np.uint8, np.uint16),
                   order='c', subdir='', verbose=2):
    dtypes = _ensure_list_or_tuple(dtypes)

    dtype_names = {np.uint8: 'uint8', np.uint16: 'uint16'}
    order_to_dir = {'c': paths.COMPRESSION_ROWMAJOR_DIR,
                    'f': paths.COMPRESSION_COLMAJOR_DIR}

    if verbose > 1:
        print "mat[:20]: ", mat[:20]

    out_paths = []
    quantize_axis = 0
    if order == 'f':
        mat = np.ascontiguousarray(mat.T)  # tofile always writes in C order
        quantize_axis = 1
        if verbose > 1:
            print "colmajor mat[:20]: ", mat[:20]

    for dtype in dtypes:
        store_mat = _quantize(mat, dtype=dtype, axis=quantize_axis)

        base_savedir = order_to_dir[order]
        savedir = os.path.join(base_savedir, dtype_names[dtype])
        if subdir:
            savedir = os.path.join(savedir, subdir)
        ensure_dir_exists(savedir)
        path = os.path.join(savedir, name + '.dat')
        store_mat.tofile(path)
        out_paths.append(path)

        if verbose > 0:
            print "saved mat {} ({}) as {}".format(name, store_mat.shape, path)

        load_mat = np.fromfile(path, dtype=dtype)

        if verbose > 1:
            print "stored mat shape: ", load_mat.shape
            print "stored mat[:20]: ", store_mat[:20]
            print "loaded mat[:20]: ", load_mat[:20]

        assert np.array_equal(store_mat.ravel(), load_mat.ravel())

        import matplotlib.pyplot as plt
        _, axes = plt.subplots(2, 2, figsize=(10, 7))
        if order == 'f':
            length = 5000
            axes[0, 0].plot(store_mat[0, :length], lw=.5)
            axes[0, 1].plot(store_mat[-1, -length:], lw=.5)
        else:
            length = 2000
            axes[0, 0].plot(store_mat.ravel()[:length], lw=.5)
            axes[0, 1].plot(store_mat.ravel()[-length:], lw=.5)
        axes[1, 0].plot(load_mat[:length], lw=.5)
        axes[1, 1].plot(load_mat[-length:], lw=.5)
        plt.show()  # upper and lower plots should be identical

    return dict(zip(dtypes, out_paths))

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

    print "mats: ", mats

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
    try:
        mats = [r.data for r in recs]  # recs is recording objects
    except AttributeError:
        mats = recs  # recs is just a data matrix (happens for UCR dataset)
        assert len(mats.shape) == 2  # fail fast if it isn't a data matrix

    return concat_and_interpolate(mats)


def main():

    # write_normal_datasets = True
    write_normal_datasets = False
    write_ucr_datasets = True
    # write_ucr_datasets = False

    STORAGE_ORDER = 'f'
    STORAGE_ORDER = 'c'

    # recordings = pamap.all_recordings()
    # # print "pamap all recording shapes: ", ['foo' for r in recordings]
    # print "pamap all recordings: ", list(recordings)
    # print "pamap all recording shapes: ", [r.data.shape for r in recordings]
    # return

    if write_normal_datasets:
        funcs_and_names = [
            # (ampds.all_gas_recordings, 'ampd_gas'),
            # (ampds.all_water_recordings, 'ampd_water'), # TODO get it working
            # (ampds.all_power_recordings, 'ampd_power'),
            (ampds.all_weather_recordings, 'ampd_weather'),
            # (uci_gas.all_recordings, 'uci_gas'),
            # (pamap.all_recordings, 'pamap'),
            # (msrc.all_recordings, 'msrc'),
        ]

        for func, name in funcs_and_names:
            recordings = func()
            print "data shapes: ", [r.data.shape for r in recordings]
            mat = mat_from_recordings(recordings)
            write_dataset(mat, name, order=STORAGE_ORDER, subdir=name, verbose=2)

    if write_ucr_datasets:
        # i = 0 # TODO rm
        # for dset in ucr.origUCRDatasets():
        for dset in ucr.allUCRDatasets():
            mat = concat_and_interpolate(dset.X)
            dtype2path = write_dataset(mat, dset.name, order=STORAGE_ORDER,
                                       subdir='ucr', verbose=2)

            # # break
            # if i == 2:
            #     print "mat shape: ", mat.shape
            #     mat_u8 = _quantize(mat, np.uint8).ravel()
            #     mat_u16 = _quantize(mat, np.uint16).ravel()

            #     # break

            #     out_mat_u8 = np.fromfile(dtype2path[np.uint8], dtype=np.uint8)
            #     out_mat_u16 = np.fromfile(dtype2path[np.uint16], dtype=np.uint16)


            #     print "mat size: ", mat_u8.size
            #     print "out_mat size: ", out_mat_u8.size
            #     print "mat[:20]: ", mat_u8[100:120]
            #     print "out_mat[:20]: ", out_mat_u8[100:120]

            #     import matplotlib.pyplot as plt; i = 0 # TODO rm
            #     _, axes = plt.subplots(2, 2, figsize=(8, 8))
            #     length = 1500
            #     axes[0, 0].plot(mat_u8[-length:])
            #     axes[0, 1].plot(mat_u16[-length:])
            #     axes[1, 0].plot(out_mat_u8[-length:])
            #     axes[1, 1].plot(out_mat_u16[-length:])
            #     plt.show()
            #     break
            # i += 1;


if __name__ == '__main__':
    _test_concat_and_interpolate()
    main()
