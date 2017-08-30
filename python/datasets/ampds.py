#!/usr/bin/env python

import os
# import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Memory

import paths
from ..utils import files
from . import viz

_memory = Memory('./')


def _list_csvs(directory):
    return files.listFilesInDir(directory, endswith='.csv', absPaths=True)


ELECTRIC_PATHS = _list_csvs(paths.AMPD2_POWER)
GAS_PATHS = _list_csvs(paths.AMPD2_GAS)
WATER_PATHS = _list_csvs(paths.AMPD2_WATER)
WEATHER_PATHS = _list_csvs(paths.AMPD2_WEATHER)

ELECTRIC_COLS = 'UNIX_TS,WHE,RSE,GRE,MHE,B1E,BME,CWE,DWE,EQE,FRE,HPE,OFE,' \
    'UTE,WOE,B2E,CDE,DNE,EBE,FGE,HTE,OUE,TVE,UNE'.split(',')

ELECTRIC_DATA_COLS = ELECTRIC_COLS[1:]
# ELECTRIC_DATA_COLS.remove('MHE')  # linear combo of other cols
# ELECTRIC_DATA_COLS.remove('UNE')  # linear combo of other cols
GAS_DATA_COLS = ['counter', 'avg_rate', 'inst_rate']
WATER_DATA_COLS = ['counter', 'avg_rate', 'inst_rate']


WEATHER_TIME_COL = 'Date/Time'
WEATHER_DATA_COLS = ['Temp (C)', 'Dew Point Temp (C)', 'Rel Hum (%)',
                     'Wind Dir (10s deg)', 'Wind Spd (km/h)',
                     'Visibility (km)', 'Stn Press (kPa)']
WEATHER_ALL_COLS = [WEATHER_TIME_COL] + WEATHER_DATA_COLS

FIG_SAVE_DIR = os.path.join('figs', 'ampds')


# ================================================================ public

class HouseRecording(object):

    def __init__(self, path, cols=None):
        data = _read_file(path)
        self.path = path
        self.name = os.path.basename(path).split('.')[0]
        self.col_names = cols
        self.sampleTimes = data[:, 0]
        self.data = data[:, 1:]  # XXX have to use all cols after the first

        # hack to deal with DWW water not having inst_rate
        self.col_names = self.col_names[:self.data.shape[1]]


class WeatherRecording(object):

    def __init__(self):
        df = _load_weather_data()
        self.name = 'weather'
        self.col_names = WEATHER_DATA_COLS
        self.sampleTimes = _datetime_strs_to_unix_timestamps(df[WEATHER_TIME_COL])
        self.data = df[WEATHER_DATA_COLS].values.astype(np.float32)


# ------------------------ top-level data loading functions

def all_power_recordings():
    # print "electric file paths: ", ELECTRIC_PATHS
    # import sys; sys.exit()
    return [HouseRecording(path, cols=ELECTRIC_DATA_COLS) for path in ELECTRIC_PATHS]


def all_gas_recordings():
    return [HouseRecording(path, cols=GAS_DATA_COLS) for path in GAS_PATHS]


def all_water_recordings():
    return [HouseRecording(path, cols=WATER_DATA_COLS) for path in WATER_PATHS]


def all_weather_recordings():
    return [WeatherRecording()]  # just one data file, so just one recording


# ================================================================ private

# def _read_file(path, cols=None):
@_memory.cache
def _read_file(path):
    df = pd.read_csv(path).fillna(method='backfill')  # hold prev val
    # if cols is not None and len(cols) > 0:
    #     timestamps = df[df.columns[0]]
    # return df.values.astype(np.int32)
    return df.values.astype(np.float32)


@_memory.cache
def _load_weather_data():
    path = WEATHER_PATHS[0]
    df = pd.read_csv(path, sep=',').fillna(method='backfill')  # hold prev val
    return df[WEATHER_ALL_COLS]


def _datetimes_to_unix_timestamps(datetimes):
    # https://stackoverflow.com/q/34038273
    return (datetimes.astype(np.int64) / 1e6).astype(np.uint64)


def _datetime_strs_to_unix_timestamps(strs):
    return _datetimes_to_unix_timestamps(pd.to_datetime(strs))


# ================================================================ main


def main():
    recordings = []
    recordings += all_gas_recordings()
    recordings += all_water_recordings()
    recordings += all_power_recordings()
    recordings += all_weather_recordings()

    norm_means = False
    # norm_means = True
    mins_zero = True

    viz.plot_recordings(recordings, norm_means=norm_means, mins_zero=mins_zero,
                        savedir=FIG_SAVE_DIR)
    # plt.show()


if __name__ == '__main__':
    main()
