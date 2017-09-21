#!/usr/bin/env python

from __future__ import division

import os
import numpy as np
from joblib import Memory

from ..utils.files import ensure_dir_exists

from . import ampds, ucr, pamap, uci_gas, msrc
from . import paths

from .. import compress

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
    mat /= max(1, np.max(mat, axis=axis, keepdims=True))
    if dtype == np.uint8:
        max_val = 255
    elif dtype in (np.uint16, np.uint32):  # NOTE: u32 -> 16 leading 0s
        max_val = 65535
    # elif dtype == np.uint32:
        # max_val == (1 << 32) - 1
    else:
        raise ValueError("Invalid dtype '{}'".format(dtype))

    return (mat * max_val).astype(dtype)


def _ensure_list_or_tuple(x):
    if not isinstance(x, (list, tuple)):
        return [x]
    return x


# want to be able to write:
#   u8 as u8
#   u16 as u16
#   u8 as u32
#   u16 as u32
#   delta [zigzag] u8 as u32
#   delta [zigzag] u16 as u32
#
#   more compactly:
#       {delta, delta+zigzag, raw} {u8, u16} as {same type, u32}

# def write_dataset(mat, name, dtypes=(np.uint8, np.uint16),
def write_dataset(mat, name, dtype, store_as_dtype=None, order='f',
                  delta_encode=False, zigzag_encode=False, subdir='', verbose=1):

    dtype_names = {np.uint8: 'uint8', np.uint16: 'uint16', np.uint32: 'uint32',
                   np.int8: 'int8', np.int16: 'int16', np.int32: 'int32'}
    utype_to_stype = {np.uint8: np.int8, np.uint16: np.int16,
                      np.uint32: np.int32}
    order_to_dir = {'c': paths.COMPRESSION_ROWMAJOR_DIR,
                    'f': paths.COMPRESSION_COLMAJOR_DIR}

    store_mat = _quantize(mat, dtype=dtype)

    if delta_encode:
        store_mat = store_mat.astype(np.int32)
        store_mat[1:, :] = store_mat[1:, :] - store_mat[:-1, :]
        if not zigzag_encode:
            dtype = utype_to_stype[dtype]
    if zigzag_encode:
        store_mat = compress.zigzag_encode(store_mat)

    # construct path at which to write data based on params
    base_savedir = order_to_dir[order]
    if store_as_dtype is None:
        child_dir = dtype_names[dtype]
    else:
        child_dir = '{}-as-{}'.format(
            dtype_names[dtype], dtype_names[store_as_dtype])
    if delta_encode:
        child_dir += '_deltas'
    if zigzag_encode:
        child_dir += '_zigzag'
    savedir = os.path.join(base_savedir, child_dir)
    if subdir:
        savedir = os.path.join(savedir, subdir)
    ensure_dir_exists(savedir)
    path = os.path.join(savedir, name + '.dat')

    # actually write out the data
    store_as_dtype = dtype if (store_as_dtype is None) else store_as_dtype
    store_mat = store_mat.astype(store_as_dtype)
    if order == 'f':
        store_mat = np.ascontiguousarray(store_mat.T)  # tofile always writes in C order
    store_mat.tofile(path)
    # out_paths.append(path)

    if verbose > 0:
        print "saved mat {} ({}) as {}".format(name, store_mat.shape, path)

    load_mat = np.fromfile(path, dtype=store_as_dtype)

    if verbose > 1:
        print "stored mat shape: ", load_mat.shape
        print "stored mat[:10]: ", store_mat[:10]
        print "loaded mat[:10]: ", load_mat[:10]

    assert np.array_equal(store_mat.ravel(), load_mat.ravel())

    if verbose > 1:
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
        # for ax in axes.ravel():
        #     ax.set_ylim([0, 255 if dtype == np.uint8 else 65535])
        plt.show()  # upper and lower plots should be identical

    # return dict(zip(dtypes, out_paths))
    return dict(dtype=path)


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
    write_normal_datasets = False
    write_ucr_datasets = False
    # write_normal_datasets = True
    write_ucr_datasets = True

    # delta_encode = False
    # zigzag_encode = False
    delta_encode = True
    zigzag_encode = True

    # STORAGE_ORDER = 'f'
    STORAGE_ORDER = 'c'

    dtypes = [np.uint8, np.uint16]
    # dtypes = [np.uint32]

    # store_as_dtype = None
    store_as_dtype = np.uint32

    if write_normal_datasets:
        funcs_and_names = [
            (ampds.all_gas_recordings, 'ampd_gas'),
            (ampds.all_water_recordings, 'ampd_water'),
            (ampds.all_power_recordings, 'ampd_power'),
            (ampds.all_weather_recordings, 'ampd_weather'),
            (uci_gas.all_recordings, 'uci_gas'),
            (pamap.all_recordings, 'pamap'),
            (msrc.all_recordings, 'msrc'),
        ]

        for func, name in funcs_and_names:
            recordings = func()
            print "data shapes: ", [r.data.shape for r in recordings]
            mat = mat_from_recordings(recordings)
            for dtype in dtypes:
                write_dataset(mat, name, order=STORAGE_ORDER, subdir=name,
                              dtype=dtype, delta_encode=delta_encode,
                              zigzag_encode=zigzag_encode,
                              store_as_dtype=store_as_dtype, verbose=1)

    if write_ucr_datasets:
        # i = 0 # TODO rm
        # for dset in ucr.origUCRDatasets():
        # for dset in list(ucr.allUCRDatasets())[:2]:
        for dset in ucr.allUCRDatasets():
            mat = concat_and_interpolate(dset.X)
            for dtype in dtypes:
                dtype2path = write_dataset(
                    mat, dset.name, order=STORAGE_ORDER, dtype=dtype,
                    subdir='ucr', delta_encode=delta_encode,
                    zigzag_encode=zigzag_encode, store_as_dtype=store_as_dtype,
                    verbose=1)

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
