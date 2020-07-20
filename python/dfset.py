#!/usr/bin/env python

import abc
from distutils.dir_util import copy_tree  # shutil.copytree fails if dir exists
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.feather as pf

from python import simple_dataframe as sdf
from python import utils  # for pd-compatible array comparisons


# TODO composition over inheritance for all the different formats; really just
# need BaseDfSet to have some object that holds some configuration and knows
# how to read and write files


class BaseDfSet(abc.ABC):

    def __init__(self, dfsdir, filetype='feather',
                 read_kwargs=None, write_kwargs=None,
                 convert_slash_to='||', verbose=0):
        self._dfsdir = dfsdir
        assert dfsdir  # can't be empty or none
        assert filetype in ('csv', 'npy', 'parquet', 'feather', 'h5', 'smart')
        self._filetype = filetype
        self._read_kwargs = read_kwargs or {}
        self._write_kwargs = write_kwargs or {}
        self._endswith = '.' + filetype
        self._ids = None
        self._convert_slash_to = convert_slash_to
        self.verbose = verbose

        if not os.path.exists(self._dfsdir):
            os.makedirs(self._dfsdir)

    def _rm_endswith(self, fname):
        if self._endswith and fname.endswith(self._endswith):
            fname = fname[:-len(self._endswith)]
        return fname

    def _id_from_dirname(self, dirname):
        return dirname
        # print(f"id = {self._rm_endswith(fname)} from dirname {fname}")
        # return self._rm_endswith(fname)

    def _colname_from_fname(self, fname):
        fname = self._rm_endswith(fname)
        return fname.replace(self._convert_slash_to, '/')

    def _basename_for_col(self, col):
        return col.replace('/', self._convert_slash_to)

    def _path(self, dfid, col=None):
        if col is None:
            return os.path.join(self._dfsdir, dfid)
        fname = self._basename_for_col(col) + self._endswith
        return os.path.join(self._dfsdir, dfid, fname)

    def _find_ids(self):
        return [self._id_from_dirname(dirname)
                for dirname in os.listdir(self._dfsdir)
                if os.path.isdir(os.path.join(self._dfsdir, dirname))]

    @property
    def filetype(self):
        return self._filetype

    @property
    def ids(self):
        if self._ids is None or len(self._ids) == 0:
            self._ids = self._find_ids()
        return self._ids

    @property
    def dir(self):
        return self._dfsdir

    # def _all_paths(self):
    #     ret = []
    #     for dfid in self.ids:
    #         idpath = self._path(dfid)
    #         ret += [os.path.join(idpath, fname)
    #                 for fname in os.listdir(idpath)]
    #     return ret

    def __len__(self):
        return len(self.ids)

    def copy_from_csvs_dir(self, dirpath, endswith='.csv', dtypes=None,
                           **read_kwargs):
        fnames = [fname for fname in os.listdir(dirpath)
                  if fname.endswith(endswith)]
        for fname in fnames:
            path = os.path.join(dirpath, fname)
            df = pd.read_csv(path, dtype=dtypes, **read_kwargs)
            if fname.endswith('.csv'):
                fname = fname[:-4]
            self[self._id_from_dirname(fname)] = df
            # print("df fname:", fname)
            # print("df dtypes:\n", df.dtypes)
            # df_hat = self[self._id_from_dirname(fname)]
            # print("df hat fname:", fname)
            # print("df hat dtypes:\n", df.dtypes)  # yep, same
        return self

    # TODO put in some caching logic so we don't actually have to touch
    # the filesystem each time

    def _cols_stored_for_dfid(self, dfid):
        dirname = self._path(dfid)
        if not os.path.exists(dirname):
            return []
        return [self._colname_from_fname(fname)
                for fname in os.listdir(dirname)
                # if fname.endswith(self._endswith)]
                if self._rm_endswith(fname) != fname]

    # def __getitem__(self, dfid, cols=None):
    def __getitem__(self, dfid_and_maybe_cols):
        if isinstance(dfid_and_maybe_cols, tuple):
            dfid, cols = dfid_and_maybe_cols   # 2 args
        else:
            dfid = dfid_and_maybe_cols
            cols = None

        if cols is None:
            cols = self._cols_stored_for_dfid(dfid)

        # ret = {}
        ret = sdf.SimpleDataFrame()
        just_one_col = False
        if isinstance(cols, str):
            cols = [cols]  # was actually just one col
            just_one_col = True

        # print("getitem() cols: ", cols)

        for col in cols:
            # path = self._path(dfid, col)

            # print("getitem: dfid, col:", dfid, col)

            # if not os.path.exists(path):
            #     print("tried to access path that doens't exist: ")
            #     # TODO add option to throw here if path not found
            #     continue  # just omit instead of crashing
            # vals = self._read_col_from_path(path)

            # if col == 'accel_valid':
            #     print("getitem: accel_valid: ", vals, type(vals), vals.dtype)

            vals = self._read_col(dfid, col)
            if vals is None:
                continue

            if just_one_col:
                # return sdf.SimpleSeries(vals)
                return pd.Series(vals)
                # return vals
            ret[col] = vals
        # if 'accel_valid' in ret:
        #     print("getitem: accel_valid info", type(ret['accel_valid']), ret['accel_valid'].dtype)
        return ret
        # return pd.DataFrame.from_dict(ret)
        # return sdf.SimpleDataFrame.from_dict(ret)

    # def __setitem__(self, dfid_and_maybe_cols, df_or_array):
    #     if isinstance(dfid_and_maybe_cols, tuple):
    #         dfid, cols = dfid_and_maybe_cols   # 2 args
    #     else:
    #         dfid = dfid_and_maybe_cols
    #         cols = None
    #     if cols is None:
    #         cols = df.columns

    def _read_col(self, dfid, col):
        path = self._path(dfid, col)
        if not os.path.exists(path):
            return None
        return self._read_col_from_path(path)

    def _normalize_idx_and_val(self, dfid_and_maybe_cols, df_or_array):
        df = df_or_array
        if isinstance(dfid_and_maybe_cols, tuple):
            assert len(dfid_and_maybe_cols) == 2  # dfid and cols
            dfid, cols = dfid_and_maybe_cols
            # wipe_others = False

            if isinstance(cols, str):
                cols = [cols]  # just a single col
                if not hasattr(df_or_array, 'columns'):
                    tmp = sdf.SimpleDataFrame()
                    tmp[cols[0]] = df_or_array
                    df = tmp
            else:
                # multiple columns passed, so df needs to actually
                # be a df and not just an array or series
                assert set(cols) == set(df.columns)

            wipe_cols = []
        else:
            dfid = dfid_and_maybe_cols
            cols = df.columns  # no cols passed, so requires df, not array
            # wipe_others = True
            wipe_cols = list(
                (set(self._cols_stored_for_dfid(dfid)) - set(cols)))

        assert list(df.columns) == list(cols)
        assert (set(wipe_cols) & set(cols)) == set()

        return dfid, cols, wipe_cols, df

    def __setitem__(self, dfid_and_maybe_cols, df_or_array):
        dfid, cols, wipe_cols, df = self._normalize_idx_and_val(
            dfid_and_maybe_cols, df_or_array)

        dfid_path = self._path(dfid)
        if not os.path.exists(dfid_path):
            os.mkdir(dfid_path)
        for col in cols:
            # don't extract the numpy array because the series might have
            # dtype that numpy lacks (in particular, the nullable int and
            # string types)
            vals = df[col]

            # if self.verbose > -1:  # TODO rm
            if self.verbose > 0:
                # print("dfid, col, vals type, vals dtype: ",
                #     dfid, col, type(vals), vals.dtype)
                print("setitem: dfid, col, dtype: ", dfid, col, vals.dtype)

            # print("setting dfid, col, vals: ", dfid, col, vals, vals.dtype)
            self._write_col_to_path(self._path(dfid, col), vals)

            # # TODO rm
            # print("vals:\n", vals)
            # vals_hat = self._read_col_from_path(self._path(dfid, col))
            # print("vals_hat:\n", vals_hat)
            # print("orig dtype, retrieved dtype: ", vals.dtype, vals_hat.dtype)
            # assert vals_hat.dtype == vals.dtype

        # if we got dfs[dfid] = df, wipe all cols not in df
        self.remove(dfid, wipe_cols)

    def remove(self, dfid, cols=None):
        if cols is None:
            os.rmdir(self._path(dfid))
            return
        if isinstance(cols, str):
            cols = [cols]
        for col in cols:
            self._remove_col(dfid, col)

    def _remove_col(self, dfid, col):
        os.remove(self._path(dfid, col))

    @abc.abstractmethod
    def _read_col_from_path(self, path):
        pass

    @abc.abstractmethod
    def _write_col_to_path(self, path, values):
        pass

    def file_sizes(self):
        all_dfids = []
        all_cols = []
        all_sizes = []
        # for path in self._all_paths():
        for dfid in self.ids:
            # print("reading sizes of id: ", dfid)
            cols = self._cols_stored_for_dfid(dfid)
            # print("cols: ", cols)
            # import sys; sys.exit()
            for col in cols:
                # print("reading size of col: ", col)
                sz = self._get_size(dfid, col)
                all_dfids.append(dfid)
                all_cols.append(col)
                all_sizes.append(sz)
        return pd.DataFrame.from_dict({
            'dfid': all_dfids, 'col': all_cols, 'nbytes': all_sizes})

    def _get_size(self, dfid, col):
        path = self._path(dfid, col)
        return os.path.getsize(path)

    def equals(self, other, raise_error=False):
        our_ids = self.ids
        other_ids = other.ids
        try:
            assert len(our_ids) == len(other_ids)
            assert np.all(np.sort(our_ids) == np.sort(other_ids))

            for dfid in our_ids:
                our_cols = self._cols_stored_for_dfid(dfid)
                other_cols = other._cols_stored_for_dfid(dfid)
                assert len(our_cols) == len(other_cols)
                assert np.all(np.sort(our_cols) == np.sort(other_cols))
                for col in our_cols:
                    our_vals = self[dfid, col]
                    other_vals = other[dfid, col]

                    assert utils.allclose(our_vals, other_vals)

                    # assert len(our_vals) == len(other_vals)
                    # # print("col: ", col)
                    # # if col == 'lon':
                    # # if col == 'c':
                    # # if col == 'gyro_valid':
                    # #     print("dfid, col = ", dfid, col)
                    # #     print("our vals:", our_vals)
                    # #     print("other vals:", other_vals)
                    # # noteq = our_vals != other_vals
                    # # print("number of neq vals: ", noteq.sum())
                    # # print("some not eq vals: ")
                    # # print(our_vals[noteq][:10], other_vals[noteq][:10])
                    # # print("some not eq idxs: ")
                    # # print(np.arange(len(our_vals))[noteq][:10])

                    # our_mask = pd.notna(our_vals)
                    # other_mask = pd.notna(other_vals)
                    # # if col == 'lon':
                    # #     print("dfid, col = ", dfid, col)
                    # #     print("our vals:", our_vals)
                    # #     print("other vals:", other_vals)
                    # assert np.array_equal(our_mask, other_mask)
                    # our_nonnans = our_vals[our_mask]
                    # other_nonnans = other_vals[other_mask]
                    # # assert np.array_equal(our_nonnans, other_nonnans)
                    # assert np.allclose(our_nonnans, other_nonnans)

                    # assert np.allclose(our_vals, other_vals, equal_nan=True)
                    # assert np.allclose(
                    #     np.sort(our_vals), np.sort(other_vals))
        except AssertionError as e:
            if raise_error:
                raise e
            return False

        return True

    def clone_into(self, other):
        # assert len(self.ids) == 1000 # TODO rm
        for i, dfid in enumerate(self.ids):
            print(f"cloning dfid: '{dfid}' ({i+1}/{len(self.ids)})")
            other[dfid] = self[dfid]
            # print(f"cloning dfid: '{dfid}'")
            # import time; time.sleep(1)
            # for col in self._cols_stored_for_dfid(dfid):
            #     print(f"col: '{col}'")
            #     other[dfid, col] = self[dfid, col]

    def copy(self, dfsdir, clobberdir=True, **kwargs):
        nokwargs = not kwargs
        kwargs.setdefault('filetype', self.filetype)
        kwargs.setdefault('convert_slash_to', self._convert_slash_to)
        if kwargs['filetype'] == self.filetype:
            kwargs.setdefault('read_kwargs', self._read_kwargs.copy())
            kwargs.setdefault('write_kwargs', self._write_kwargs.copy())

        # TODO have a well-defined params() method instead of just
        # looking at which args in init need to be copied

        # print(f"copying from {self.dir} to {dfsdir}")

        if nokwargs:  # same format, so can just copy files
            if clobberdir and os.path.exists(dfsdir):
                shutil.rmtree(dfsdir)
            # this line needs python 3.8 to get the dirs_exist_ok param
            # shutil.copytree(self.dir, dfsdir, dirs_exist_ok=not clobberdir)
            copy_tree(self.dir, dfsdir)
            return make_dfset(dfsdir=dfsdir, **kwargs)

        other = make_dfset(dfsdir=dfsdir, **kwargs)
        self.clone_into(other)  # maybe different fmt, so need to read + write
        return other


class CsvDfSet(BaseDfSet):
    # XXX only supports float, since doesn't preserve dtype

    def __init__(self, *args, **kwargs):
        kwargs['filetype'] = 'csv'
        super().__init__(*args, **kwargs)
        self._read_kwargs.setdefault('delimiter', ',')
        self._write_kwargs.setdefault('delimiter', ',')
        self._write_kwargs.setdefault('fmt', '%g')  # default is %.18e

    def _read_col_from_path(self, path):
        # print("reading from path: ", path)
        # with open(path, 'r') as f:
        #     print("contents:\n", f.read())
        return np.loadtxt(path, **self._read_kwargs)

    def _write_col_to_path(self, path, values):
        # print("writing to path: ", path)
        np.savetxt(path, np.asarray(values), **self._write_kwargs)


class NpyDfSet(BaseDfSet):

    def __init__(self, *args, **kwargs):
        kwargs['filetype'] = 'npy'
        super().__init__(*args, **kwargs)
        # pickle can run into compatibility issues
        # self._write_kwargs.setdefault('allow_pickle', False)

    def _read_col_from_path(self, path):
        # print("reading from path: ", path)
        return np.load(path, **self._read_kwargs)

    def _write_col_to_path(self, path, values):
        # print("writing to path: ", path)
        np.save(path, np.asarray(values), **self._write_kwargs)


class FeatherDfSet(BaseDfSet):

    def __init__(self, *args, **kwargs):
        kwargs['filetype'] = 'feather'
        super().__init__(*args, **kwargs)
        self._write_kwargs.setdefault('compression', 'uncompressed')

    def _read_col_from_path(self, path):
        df = pf.read_table(path).to_pandas()
        return df[df.columns[0]]

    def _write_col_to_path(self, path, values):
        s = pd.Series(values, dtype=values.dtype)
        tbl = pa.Table.from_pandas(s.to_frame(), preserve_index=False)
        pf.write_feather(tbl, path, **self._write_kwargs)


class SmartDfSet(BaseDfSet):

    def __init__(self, *args, np_write_kwargs=None, feather_write_kwargs=None,
                 **kwargs):
        kwargs['filetype'] = 'smart'
        super().__init__(*args, **kwargs)
        self._np_write_kwargs = np_write_kwargs or {}
        self._np_write_kwargs.setdefault('allow_pickle', False)
        self._feather_write_kwargs = feather_write_kwargs or {}
        self._feather_write_kwargs.setdefault('compression', 'uncompressed')

    # override behavior that assumes fixed file extension
    # def _cols_stored_for_dfid(self, dfid):
    #     dirname = self._path(dfid)
    #     if not os.path.exists(dirname):
    #         return []
    #     ret = [self._colname_from_fname(fname)
    #            for fname in os.listdir(dirname)
    #            if self._rm_endswith(fname) != fname]
    #     return ret

    def _rm_endswith(self, fname):
        for endswith in ['.npy', '.feather', '.smart']:
            if fname.endswith(endswith):
                fname = fname[:-len(endswith)]
        return fname

    def _remove_col(self, dfid, col):
        path = self._path(dfid, col)
        for ext in ('.npy', '.feather'):
            try:
                os.remove(path + ext)
            except IOError:
                pass

    def _get_size(self, dfid, col):
        path = self._path(dfid, col)
        for ext in ('.npy', '.feather'):
            try:
                return os.path.getsize(path + ext)
            except IOError:
                pass

    def _read_col(self, dfid, col):
        # omits fast fail if path doesn't exist
        return self._read_col_from_path(self._path(dfid, col))

    def _read_col_from_path(self, path):
        # print("readcol: trying to read from path: ", path)

        # df = pf.read_table(path).to_pandas()
        np_path = path + '.npy'
        feather_path = path + '.feather'
        if os.path.exists(np_path):
            return np.load(np_path)
        elif os.path.exists(feather_path):
            df = pf.read_table(feather_path).to_pandas()
            return df[df.columns[0]]
        print("neither path existed!", np_path, feather_path)
        return None  # neither path exists

    def _write_col_to_path(self, path, values):
        try:
            _ = np.array([], dtype=values.dtype)  # throws if not numpy dtype
            vals = values.values  # pull out np array
            path += '.npy'
            np.save(path, vals, **self._np_write_kwargs)
            altpath = path + '.feather'
            if os.path.exists(altpath):
                os.remove(altpath)
        except TypeError:
            # dtype numpy can't handle; write out a pandas series instead
            tbl = pa.Table.from_pandas(values.to_frame(), preserve_index=False)
            pf.write_feather(tbl, path, **self._feather_write_kwargs)

            altpath = path + '.npy'
            if os.path.exists(altpath):
                os.remove(altpath)


class ParquetDfSet(BaseDfSet):

    def __init__(self, *args, **kwargs):
        kwargs['filetype'] = 'parquet'
        super().__init__(*args, **kwargs)
        # self._write_kwargs.setdefault('compression', 'Zstd')
        # self._write_kwargs.setdefault('compression', 'gzip')
        # self._write_kwargs.setdefault('compression', 'snappy')
        self._write_kwargs.setdefault('compression', None)
        self._write_kwargs.setdefault('version', '2.0')

    def _read_col_from_path(self, path):
        # print("reading from path: ", path)
        # return pd.read_parquet(path, **self._read_kwargs)['_']
        return pq.read_table(path, columns=['_']).to_pandas()['_']

    def _write_col_to_path(self, path, values):
        s = pd.Series(values, dtype=values.dtype, name='_')
        tbl = pa.Table.from_pandas(s.to_frame(), preserve_index=False)
        pq.write_table(tbl, path, **self._write_kwargs)


class H5DfSet(BaseDfSet):
    # it would be a better use of h5 / less load on the filesystem if we put
    # everything in one h5 file; but for the purpose of assessing compression
    # ratios, having separate files for each (dfid, col) makes life way easier

    def __init__(self, *args, **kwargs):
        kwargs['filetype'] = 'h5'
        super().__init__(*args, **kwargs)
        # we're mostly using h5 as a way to get bzip2
        self._write_kwargs.setdefault('complib', 'bzip2')
        self._write_kwargs.setdefault('complevel', 9)

    def _read_col_from_path(self, path):
        # print("reading from path: ", path)
        # return pd.from_parquet(path, **self._read_kwargs)
        return pd.read_hdf(path, mode='r', **self._read_kwargs)['_']

    def _write_col_to_path(self, path, values):
        # print("writing to path: ", path)
        # df = pd.Series.from_array(values)
        df = pd.DataFrame.from_dict({'_': values})
        return df.to_hdf(path, key='_', mode='w', **self._write_kwargs)


def make_dfset(filetype='smart', dfsdir=None, csvsdir=None, dtypes=None,
               **kwargs):
    if dfsdir is None:
        dfsdir = tempfile.mkdtemp()

    if filetype == 'csv':
        dfs = CsvDfSet(dfsdir, **kwargs)
    elif filetype == 'npy':
        dfs = NpyDfSet(dfsdir, **kwargs)
    elif filetype == 'parquet':
        dfs = ParquetDfSet(dfsdir, **kwargs)
    elif filetype == 'feather':
        dfs = FeatherDfSet(dfsdir, **kwargs)
    elif filetype == 'h5':
        dfs = H5DfSet(dfsdir, **kwargs)
    elif filetype == 'smart':
        dfs = SmartDfSet(dfsdir, **kwargs)
    else:
        raise ValueError(f"unrecognized filetype: '{filetype}'")

    if csvsdir is not None:
        dfs.copy_from_csvs_dir(csvsdir, dtypes=dtypes)
    return dfs

    # def __getitem__(self, dfid, cols=None):
    #     pass # TODO

    # def __setitem__(self, dfid, df):
    #     for col in df:
    #         pass # TODO
    #         # self._write_col_to_path(self._path(dfid, col), df[col])
