#!/usr/bin/env python

# from __future__ import print_function

import collections
import os
import numpy as np
import pandas as pd

# from python.utils import files


INPUT_DATA_DIR = os.path.expanduser('~/Desktop/sample_ride_data')
OUTPUT_DATA_DIR = os.path.expanduser(
    '~/Desktop/datasets/compress/rowmajor/')
DSET_NAME = 'car_sample'

ALL_COLS = [
    'time',
    'lat',
    'lon',
    'gps_speed(m/s)',
    'gps_valid',
    'accel_lon',
    'accel_lat',
    'accel_lon_smoothed',
    'accel_lat_smoothed',
    'accel_gps',
    'accel_valid',
    'speed_limit(m/s)',
    'ax',
    'ay',
    'az',
    'accel_z',
    'accel_z_smoothed',
    'g',
    'phone_moved',
    'raw_gps_speed(m/s)',
    'mm_lat',
    'mm_lon',
    'mm_dist_km',
    'wx',
    'wy',
    'wz',
    'rwx',
    'rwy',
    'rwz',
    'screen_state',
    'distraction',
    'roll',
    'pitch',
    'yaw',
    'gps_altitude',
    'gps_heading',
    'gps_accuracy',
    'link_id',
    'speed_limit_kmh_smooth',
    'gps_valid2',
    'raw_gps_speed2(m/s)',
    'barometric_altitude',
    'barometric_road_pitch',
    'solar_altitude',
    'solar_azimuth',
    'gyro_valid',
    # tag cols may not be present
    # 'tag_lat_smoothed',
    # 'tag_lon_smoothed',
    # 'tag_vert_smoothed'
]

# ColumnInfo = collections.namedtuple(
#     'ColumnInfo', 'name min max minbits'.split())

# COLS = [
#     ColumnInfo('time', 0, None, 64),
#     ColumnInfo('lat', -90, 90, 64),
#     ColumnInfo('lon', -90, 90, 64),
#     ColumnInfo('gps_speed(m/s)',
#     'gps_valid',
#     'accel_lon',
#     'accel_lat',
#     'accel_lon_smoothed',
#     'accel_lat_smoothed',
#     'accel_gps',
#     'accel_valid',
#     'speed_limit(m/s)',
#     'ax',
#     'ay',
#     'az',
#     'accel_z',
#     'accel_z_smoothed',
#     'g',
#     'phone_moved',
#     'raw_gps_speed(m/s)',
#     'mm_lat',
#     'mm_lon',
#     'mm_dist_km',
#     'wx',
#     'wy',
#     'wz',
#     'rwx',
#     'rwy',
#     'rwz',
#     'screen_state',
#     'distraction',
#     'roll',
#     'pitch',
#     'yaw',
#     'gps_altitude',
#     'gps_heading',
#     'gps_accuracy',
#     'link_id',
#     'speed_limit_kmh_smooth',
#     'gps_valid2',
#     'raw_gps_speed2(m/s)',
#     'barometric_altitude',
#     'barometric_road_pitch',
#     'solar_altitude',
#     'solar_azimuth',
#     'gyro_valid',
# ]


I64_COLS = ['time', 'accel_valid', 'gps_valid', 'tag_valid',
            'distraction', 'link_id']

SCHEMA = {name: np.float64 for name in ALL_COLS}
SCHEMA.update({name: np.int64 for name in I64_COLS})


def data_paths():
    files = [f for f in os.listdir(INPUT_DATA_DIR) if f.endswith('.csv')]
    return sorted([os.path.join(INPUT_DATA_DIR, f) for f in files])


def data_df_from_path(path):
    return pd.read_table(path, sep=',', usecols=ALL_COLS, dtype=SCHEMA)


def dump_raw_data(df, path):
    recarray = df.to_records(index=False)
    recarray.tofile(path)
    return (recarray.shape[0], recarray.itemsize)


def quantize_dfs(dfs, how=None):
    if how is None:
        return dfs

    how2dtype = {'u8': np.uint8, 'u16': np.uint16}
    cardinalities = {
        np.uint8: 1 << 8,
        np.uint16: 1 << 16,
        np.uint32: 1 << 32,
        np.uint64: (1 << 63) + ((1 << 63) - 1)}  # can't store 1 << 64

    if how not in how2dtype:
        raise ValueError(
            "Unrecognized quantization requested: '{}'".format(how))

    new_dtype = how2dtype[how]

    df = pd.concat(dfs, axis=0)
    df.fillna(0, inplace=True)

    # print "df.dtypes", df.dtypes
    # return

    QuantizeInfo = collections.namedtuple(
        'QuantizeInfo', 'offset step dtype zero'.split())

    quantize_infos = {}

    for col, dtype in zip(df, df.dtypes):
        if dtype not in (np.float32, np.float64):
            continue  # only quantize float columns

        # print "col: ", col
        data = df[col].as_matrix()
        uniq_vals = np.unique(data).astype(np.float64)  # note these are sorted

        if len(uniq_vals) == 1:
            quantize_infos[col] = QuantizeInfo(None, None, None, zero=True)
            continue

        has_neg_vals = np.any(uniq_vals < 0)
        largest_abs_val = max(np.abs(uniq_vals[0]), np.abs(uniq_vals[-1]))

        # compute smallest quantization step that wouldn't lose
        # ability to distinguish between successive (sorted) values
        diffs = (uniq_vals[1:] - uniq_vals[:-1]).astype(np.float64)
        quantize_step = np.min(diffs)
        quantize_step = min(1, quantize_step)  # assume no larger than 1
        # if len(uniq_vals) > 2:
        #     diffs2 = uniq_vals[2:] - uniq_vals[:-2]
        #     quantize_step = min(quantize_step, np.min(diffs2) / 2)

        # print "uniq vals: ", uniq_vals[:20]
        # print "min val: ", np.min(uniq_vals)
        # assert np.abs(np.min(uniq_vals) - uniq_vals[0]) < .0001
        current_min = uniq_vals[0]
        new_min = -largest_abs_val if has_neg_vals else 0

        # ensure that spread of representable values includes all
        # observed values (this comes up if a wide range of values
        # are represented with high precision, relative to the number
        # of bits in the requested dtype; this way we lose precision
        # instead of getting clipping).
        nsteps = cardinalities[new_dtype]
        new_max = new_min + quantize_step * nsteps
        if new_max < largest_abs_val:
            quantize_step = (largest_abs_val - new_min) / nsteps

        # handle known constraints
        use_dtype = new_dtype
        if 'lat' in col:
            use_dtype = np.uint32
            new_min = -90.
            quantize_step = (180. / cardinalities[use_dtype])
        elif ('lon' in col) or (col in ('roll', 'pitch', 'yaw')):
            use_dtype = np.uint32
            new_min = -180.
            quantize_step = (360. / cardinalities[use_dtype])

        offset = new_min - current_min
        if new_min < 0:
            offset -= new_min

        quantize_infos[col] = QuantizeInfo(
            offset=offset, step=quantize_step, dtype=use_dtype, zero=False)

        # new_vals = ((data + offset) / quantize_step).astype(use_dtype)
        # df[col] = new_vals

    unmodified_cols = [col for col in df.columns if col not in quantize_infos]

    # print "quantize infos: ", quantize_infos
    for col, info in quantize_infos.items():
        if col == 'yaw':  # or True:
            print "------ {}".format(col)
            print "info: ", info
    # print ">>>>>>> unmodified_cols:", unmodified_cols

    ret = []
    for df in dfs:
        df.fillna(0, inplace=True)

        new_df = df[unmodified_cols].copy()
        # print "unmodified cols initial data"
        # print df[unmodified_cols][:10]

        # new_df = pd.DataFrame()
        # for col in unmodified_cols:
        #     new_df[col] = df[col]
        # print "new df initial types:", new_df.dtypes

        quant_losses = {}
        for col, info in quantize_infos.items():
            if info.zero:
                # df[col] = 0
                new_df[col] = np.zeros(df.shape[0], dtype=new_dtype)
                quant_losses[col] = 0
                # print "zeroing col"
                continue
            vals = df[col].as_matrix()
            quantized = ((vals + info.offset) / info.step).astype(info.dtype)
            new_df[col] = quantized
            # new_df[col] = ((df[col] + info.offset) / info.step).astype(info.dtype)

            reconstructed = (quantized.astype(np.float64) * info.step) - info.offset
            errs = vals - reconstructed
            quant_losses[col] = np.mean(errs * errs) / np.var(vals)

            # if col == 'yaw':
            #     print "min, max = ", np.min(vals), np.max(vals)
            #     print ""
            #     print "orig, shifted + scaled, quantized, reconstructed:"
            #     print vals[:20]
            #     print ((vals[:20] + info.offset) / info.step)
            #     print quantized[:20]
            #     print reconstructed[:20]
            #     print errs[:20]
            #     # print vals[-20:]
            #     # print ((vals[-20:] + info.offset) / info.step)
            #     # print quantized[-20:]
            #     # print reconstructed[-20:]

        # # compute quantization loss
        print " ================================ quantization losses:"
        for col in sorted(quant_losses):
            # if col != 'yaw': continue
            # print "{}:\n\t{}".format(col, quant_losses[col])
            print "{}, {}".format(col, quant_losses[col])

        # print "new df types: ", new_df.dtypes
        ret.append(new_df)

    return ret

            # print "new vals[:20]: ", df[col].as_matrix()[:20]

    # if how == 'u8':
    #     dtype = np.uint8
    # if how == 'u16'
    #     pass # TODO


def create_dataset(quantize=None, actually_write=True):
    paths = data_paths()
    dfs = [data_df_from_path(path) for path in paths]
    # # TODO rm
    # paths = [paths[-1]]
    # dfs = [dfs[-1]]
    dfs = quantize_dfs(dfs, how=quantize)

    # print "create_dataset: aborting after quantization"  # TODO rm
    # return

    for read_path, df in zip(paths, dfs):
        name = os.path.basename(read_path).split('.')[0]

        # df = data_df_from_path(read_path)

        subdir = 'raw'
        if quantize == 'u8':
            subdir = 'uint8'
        elif quantize == 'u16':
            subdir = 'uint16'

        # what do the deltas look like as byte streams?
        #
        # ra = df.to_records(index=False)
        # row_nbytes = ra.itemsize
        # print "row nbytes: ", row_nbytes
        # if quantize == 'u8':
        #     as_u8s = np.frombuffer(ra.data, dtype=np.uint8).reshape(-1, row_nbytes)
        #     print "u8 deltas:\n", as_u8s[1] - as_u8s[0]
        # elif quantize == 'u16':
        #     as_u16s = np.frombuffer(ra.data, dtype=np.uint16).reshape(-1, row_nbytes/2)
        #     print "u16 deltas:\n", as_u16s[1] - as_u16s[0]
        # continue

        save_dir = os.path.join(OUTPUT_DATA_DIR, subdir, DSET_NAME)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, name + '.dat')

        if actually_write:
            nrows, nbytes_per_row = dump_raw_data(df, save_path)

            print "{} -> {}".format(read_path, save_path)
            print "shape in bytes: {}, {}".format(nrows, nbytes_per_row)


def main():
    # create_dataset()
    # create_dataset(quantize='u8')
    create_dataset(quantize='u16', actually_write=False)

    # # parquet files
    # #
    # # print "writing out parquet files"
    # paths = data_paths()
    # dfs = [data_df_from_path(path) for path in paths]
    # dfs = quantize_dfs(dfs, how='u16')
    # sizes = []
    # for path, df in zip(paths, dfs):
    #     name = os.path.basename(path).split('.')[0]
    #     out_path = '{}_16b.parquet'.format(name)
    #     df.to_parquet(out_path)
    #     sizes.append(os.stat(out_path).st_size)
    # print "parquet file sizes (total = {})".format(np.sum(sizes))
    # print sizes

    # yes, looks like float deltas basically yield random numbers when
    # treated as uints
    #
    # path = data_paths()[-1]
    # ra = data_df_from_path(path).to_records(index=False)
    # as_u16s = np.frombuffer(ra.data, dtype=np.uint16).reshape(-1, 368/2)
    # as_u8s = np.frombuffer(ra.data, dtype=np.uint8).reshape(-1, 368)
    # print "u16 deltas: ", as_u16s[1] - as_u16s[0]
    # print "u8 deltas: ", as_u8s[1] - as_u8s[0]

    # # deltas = ra[1] - ra[0]
    # row0 = np.frombuffer(ra[0], dtype=np.uint8)
    # row1 = np.frombuffer(ra[1], dtype=np.uint8)
    # # print "deltas shape: ", deltas.shape
    # print "ar shape: ", row0.shape
    # print "ar shape: ", row1.shape

    # deltas = row1 - row0
    # print "deltas: ", deltas


    # paths = data_paths()
    # df = pd.read_table(paths[0], sep=',', dtype=SCHEMA)
    # # print "using path: ", paths[0]
    # print "df cols: ", df.columns
    # # df.to_csv('tmp.csv', na_rep='nan', index=False)

    # feather file
    #
    # print "writing out feather file"
    # df.to_feather('tmp.feather')

    # this appears to all be what we want
    #
    # print "recarray info:"
    # ra = df.to_records(index=False)
    # print "dtype: ", ra.dtype
    # print "itemsize (expected itemsize): ", ra.itemsize, 8 * len(ALL_COLS)
    # print "ndim: ", ra.ndim
    # print "shape: ", ra.shape
    # print "nbytes (expected nbytes): ", ra.nbytes, ra.size * ra.itemsize

    # ra.tofile('tmp.dat')


if __name__ == '__main__':
    main()
