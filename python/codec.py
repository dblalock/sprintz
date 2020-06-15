#!/usr/bin/env python

import abc
import numpy as np
import pandas as pd


class BaseCodec(abc.ABC):

    def __init__(self):
        self._needs_training = True

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


class DeltaCodec(BaseCodec):

    def encode(self, df):
        for col in df:
            vals = df[col]
            df[col][1:] = vals[1:] - vals[:-1]

    def decode(self, df, header):
        for col in df:
            df[col] = np.cumsum(df[col])



