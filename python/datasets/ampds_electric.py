#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
from joblib import Memory

import paths
from .utils import files


_memory = Memory('./')


def _list_csvs(directory):
    files.listFilesInDir(directory, endswith='.csv', absPaths=True)


ELECTRIC_PATHS = _list_csvs(paths.AMPD2_POWER)
GAS_PATHS = _list_csvs(paths.AMPD2_GAS)
WATER_PATHS = _list_csvs(paths.AMPD2_WATER)

ELECTRIC_COLS = 'UNIX_TS,WHE,RSE,GRE,MHE,B1E,BME,CWE,DWE,EQE,FRE,HPE,OFE,' \
    'UTE,WOE,B2E,CDE,DNE,EBE,FGE,HTE,OUE,TVE,UNE'.split(',')


# ================================================================
# Public
# ================================================================

class Recording(object):

    def __init__(self, path):
        data = _read_file(path)
        self.path = path
        self.name = os.path.basename(path)
        self.timestamps = data[:, 0]
        self.data = data[:, 1:]


def all_electric_recordings():
    return [Recording(path) for path in ELECTRIC_PATHS]


def all_gas_recordings():
    return [Recording(path) for path in GAS_PATHS]


def all_water_recordings():
    return [Recording(path) for path in WATER_PATHS]

# ================================================================
# Private
# ================================================================

@_memory.cache
def _read_file(path):
    df = pd.read_csv(path)
    return df.values.astype(np.int32)


def main():
    import matplotlib.pyplot as plt
    for r in all_recordings():
        print "recording {} has data of shape {}".format(r.name, r.data.shape)
        _, axes = plt.subplots(2)
        axes[0].plot(r.timestamps[:1000], r.data[:1000])
        axes[1].plot(r.timestamps[-1000:], r.data[-1000:])
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
