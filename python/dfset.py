#!/usr/bin/env python

import abc
from distutils.dir_util import copy_tree  # shutil.copytree fails if dir exists
import os
import shutil
import tempfile


import numpy as np
import pandas as pd

from python import simple_dataframe as sdf


class BaseDfSet(abc.ABC):

    def __init__(self, dfsdir, filetype='csv',
                 read_kwargs=None, write_kwargs=None,
                 convert_slash_to='||', verbose=0):
        self._dfsdir = dfsdir
        assert dfsdir  # can't be empty or none
        assert filetype in ('csv', 'npy', 'parquet', 'h5')
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

    def _id_from_dirname(self, fname):
        # print(f"id = {self._rm_endswith(fname)} from dirname {fname}")
        return self._rm_endswith(fname)

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
        return [self._id_from_dirname(fname)
                for fname in os.listdir(self._dfsdir)
                if os.path.isdir(os.path.join(self._dfsdir, fname))]

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


    def _all_paths(self):
        ret = []
        for dfid in self.ids:
            idpath = self._path(dfid)
            ret += [os.path.join(idpath, fname)
                    for fname in os.listdir(idpath)]
        return ret

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
                if fname.endswith(self._endswith)]

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
            path = self._path(dfid, col)
            if not os.path.exists(path):
                # TODO add option to throw here if path not found
                continue  # just omit instead of crashing
            vals = self._read_col_from_path(path)

            # if col == 'accel_valid':
            #     print("getitem: accel_valid: ", vals, type(vals), vals.dtype)

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

            if self.verbose > 0:
                # print("dfid, col, vals type, vals dtype: ",
                #     dfid, col, type(vals), vals.dtype)
                print("dfid, col, vals dtype: ", dfid, col, vals.dtype)

            # print("setting dfid, col, vals: ", dfid, col, vals, vals.dtype)
            self._write_col_to_path(self._path(dfid, col), vals)

        # if we got dfs[dfid] = df, wipe all cols not in df
        self.remove(dfid, wipe_cols)

    def remove(self, dfid, cols=None):
        if cols is None:
            os.rmdir(self._path(dfid))
            return
        if isinstance(cols, str):
            cols = [cols]
        for col in cols:
            self._remove_col_at_path(self._path(dfid, col))

    def _remove_col_at_path(self, path):
        os.remove(path)  # separate method in case not one file <=> one col

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
                path = self._path(dfid, col)
                sz = os.path.getsize(path)
                all_dfids.append(dfid)
                all_cols.append(col)
                all_sizes.append(sz)
        return pd.DataFrame.from_dict({
            'dfid': all_dfids, 'col': all_cols, 'nbytes': all_sizes})

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
                    assert len(our_vals) == len(other_vals)
                    # print("col: ", col)
                    # if col == 'lon':
                    # if col == 'c':
                    #     print("dfid, col = ", dfid, col)
                    #     print("our vals:", our_vals)
                    #     print("other vals:", other_vals)
                    # noteq = our_vals != other_vals
                    # print("number of neq vals: ", noteq.sum())
                    # print("some not eq vals: ")
                    # print(our_vals[noteq][:10], other_vals[noteq][:10])
                    # print("some not eq idxs: ")
                    # print(np.arange(len(our_vals))[noteq][:10])

                    our_mask = pd.notna(our_vals)
                    other_mask = pd.notna(other_vals)
                    # if col == 'lon':
                    #     print("dfid, col = ", dfid, col)
                    #     print("our vals:", our_vals)
                    #     print("other vals:", other_vals)
                    assert np.array_equal(our_mask, other_mask)
                    our_nonnans = our_vals[our_mask]
                    other_nonnans = other_vals[other_mask]
                    # assert np.array_equal(our_nonnans, other_nonnans)
                    assert np.allclose(our_nonnans, other_nonnans)

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

    # def nfiles(self):
    #     return len(self._all_paths())

    # def __getitem__(self, dfid, cols=None):
    #     path = self._path_from_id(dfid)
    #     if self._filetype == 'csv':
    #         df = pd.read_csv(path, **self._read_kwargs)
    #         if cols is not None:
    #             return df[cols]
    #         return df
    #     elif self._filetype == 'parquet':
    #         return pd.read_parquet(path, columns=cols, **self._read_kwargs)


class CsvDfSet(BaseDfSet):
    # XXX only supports float, since doesn't preserve dtype

    def __init__(self, *args, **kwargs):
        kwargs['filetype'] = 'csv'
        super().__init__(*args, **kwargs)
        self._read_kwargs.setdefault('delimiter', ',')
        self._write_kwargs.setdefault('delimiter', ',')

    def _read_col_from_path(self, path):
        # print("reading from path: ", path)
        return np.loadtxt(path, **self._read_kwargs)

    def _write_col_to_path(self, path, values):
        # print("writing to path: ", path)
        np.savetxt(path, np.asarray(values), **self._write_kwargs)


class NpyDfSet(BaseDfSet):

    def __init__(self, *args, **kwargs):
        kwargs['filetype'] = 'npy'
        super().__init__(*args, **kwargs)
        # compression is a good idea, but messes up file size measurements
        # self._write_kwargs.setdefault('compression', None)

    def _read_col_from_path(self, path):
        # print("reading from path: ", path)
        return np.load(path, **self._read_kwargs)

    def _write_col_to_path(self, path, values):
        # print("writing to path: ", path)
        np.save(path, np.asarray(values), **self._write_kwargs)


class ParquetDfSet(BaseDfSet):

    def __init__(self, *args, **kwargs):
        kwargs['filetype'] = 'parquet'
        super().__init__(*args, **kwargs)
        # self._write_kwargs.setdefault('compression', 'Zstd')
        # self._write_kwargs.setdefault('compression', 'gzip')
        # self._write_kwargs.setdefault('compression', 'snappy')
        self._write_kwargs.setdefault('compression', None)
        self._write_kwargs.setdefault('index', False)

    def _read_col_from_path(self, path):
        # print("reading from path: ", path)
        return pd.read_parquet(path, **self._read_kwargs)['_']

    def _write_col_to_path(self, path, values):
        # # print("parquet write: path =", path)
        # # # df = pd.Series.from_array(values)
        # # # df = pd.DataFrame([pd.Series.from_array(values)])
        # print("parquet write: values shape", values.shape)
        # print("parquet write: values type", type(values))
        # # # values.foo()
        # # # assert hasattr(values, 'dtype')
        # print("parquet write: values dtype", values.dtype)
        # # # print("parquet write: uniq values", np.unique(values))
        # # # print("parqetdfset: writing values:\n", values, type(values))
        # print("parqetdfset: writing head values:\n", values[:20])
        # print("parqetdfset: writing tail values:\n", values[-20:])

        # try:
        #     df = pd.DataFrame.from_dict({'_': values})
        # except ValueError:  # happens if values is a
        s = pd.Series(values, dtype=values.dtype, name='_')
        s.to_frame().to_parquet(path, **self._write_kwargs)
        # print("s shape: ", s.shape)
        # df = s.to_frame()
        # print("df shape: ", df.shape)
        # df = pd.DataFrame.from_dict({'_': values}, dtype=values.dtype)
        # df = pd.DataFrame.from_dict({'_': s})
        # return df.to_parquet(path, **self._write_kwargs)


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


def make_dfset(filetype, dfsdir=None, csvsdir=None, dtypes=None, **kwargs):
    if dfsdir is None:
        dfsdir = tempfile.mkdtemp()

    if filetype == 'csv':
        dfs = CsvDfSet(dfsdir, **kwargs)
    elif filetype == 'npy':
        dfs = NpyDfSet(dfsdir, **kwargs)
    elif filetype == 'parquet':
        dfs = ParquetDfSet(dfsdir, **kwargs)
    elif filetype == 'h5':
        dfs = H5DfSet(dfsdir, **kwargs)
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
