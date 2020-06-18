#!/usr/bin/env python

import abc
import numpy as np


def _wrap_in_list_if_str(obj):
    return ([obj] if isinstance(obj, str) else obj)


class BaseCodec(abc.ABC):

    def __init__(self):
        self._needs_training = False

    # @property
    def cols(self):
        readonly_cols = self.readonly_cols() or []
        write_cols = self.write_cols() or []
        return (readonly_cols + write_cols) or None

    # @property
    def readonly_cols(self):
        return None  # None = all of them

    # @abc.abstractmethod
    # @property
    def write_cols(self):
        return None  # None = all of them

    def train_cols(self):
        return self.cols()

    def train(self, dfc):
        pass

    @property
    def needs_training(self):
        return self._needs_training

    @needs_training.setter
    def needs_training(self, val):
        self._needs_training = val

    # TODO using json-serializable params would be less brittle
    # than pickling in case the class definition changes between
    # serialization and deserialization
    #
    # # @abc.abstractmethod
    # # # @property
    # def params(self):
    #     pass
    #
    # @classmethod
    # def from_params(self, params):
    #     pass

    @abc.abstractmethod
    def encode(self, df):
        pass

    @abc.abstractmethod
    def decode(self, df, header):
        pass


class Debug(BaseCodec):

    def encode(self, df):
        # return df.columns, None # TODO rm after debug
        for col in df:
            df[col] = df[col].values[::-1]
            # df[col] -= 1
        return df, None

    def decode(self, df, header_unused):
        # return # TODO rm after debug
        for col in df:
            df[col] = df[col].values[::-1]
            # df[col] += 1
        # print("df col a within decode: ", df['a'])
        return df


class Delta(BaseCodec):

    def __init__(self, which_cols=None):
        super().__init__()
        self._cols = _wrap_in_list_if_str(which_cols)  # None = all
        # self.cols_to_sum = _wrap_in_list_if_str(cols_to_sum)
        # self.col_to_predict = col_to_predict

    def encode(self, df):
        # print("running encode!")
        # return df.columns, None # TODO rm after debug
        use_cols = self._cols if self._cols is not None else df.columns
        for col in use_cols:
            vals = df[col].values
            vals[1:] -= vals[:-1]
            df[col] = vals
            # df[col] = df[col].values[::-1]  # TODO rm after debug
        return df[use_cols], None

    def decode(self, df, header_unused):
        # print("running decode!")
        # return # TODO rm after debug
        use_cols = self._cols if self._cols is not None else df.columns
        for col in use_cols:
            df[col] = np.cumsum(df[col])
            # df[col] = df[col].values[::-1]  # TODO rm after debug
        return df[use_cols]


class ColSumPredictor(BaseCodec):
    """predicts one column as the sum of one or more other columns"""

    def __init__(self, cols_to_sum, col_to_predict):
        super().__init__()
        self.cols_to_sum = _wrap_in_list_if_str(cols_to_sum)
        self.col_to_predict = col_to_predict

    def readonly_cols(self):
        return self.cols_to_sum

    def write_cols(self):
        return [self.col_to_predict]

    def _compute_predictions(self, df):
        predictions = df[self.cols_to_sum[0]]
        if len(self.cols_to_sum) > 1:
            for col in self.cols_to_sum[1:]:
                predictions += df[col].values
        return predictions

    def encode(self, df):
        predictions = self._compute_predictions(df)
        df[self.col_to_predict] = df[self.col_to_predict].values - predictions
        return df[self.col_to_predict], None  # no headers

    def decode(self, df):
        predictions = self._compute_predictions(df)
        df[self.col_to_predict] = df[self.col_to_predict].values + predictions
        return df[self.col_to_predict]
