#!/usr/bin/env python

import os
import numpy as np
from joblib import Memory

from . import viz
from . import paths

_memory = Memory('./')

DATA_DIR = paths.UCI_GAS
CO_PATH = os.path.join(DATA_DIR, 'ethylene_CO.txt')
METHANE_PATH = os.path.join(DATA_DIR, 'ethylene_methane.txt')

COLS = ['time', 'gas_concentration', 'ethyl_concentration']
COLS += ['sensor_type0_{}'.format(i) for i in range(8)]
COLS += ['sensor_type1_{}'.format(i) for i in range(8)]

FIG_SAVE_DIR = os.path.join('figs', 'uci_gas')


# ================================================================
# Public
# ================================================================

class Recording(object):

    def __init__(self, path):
        # assert path in (CO_PATH, METHANE_PATH)
        data = _read_gas_file(path)
        self.path = path
        self.name = os.path.basename(path).split('.')[0]
        self.sampleTimes = data[:, 0]
        self.gas_concentration = data[:, 1]
        self.ethyl_concentration = data[:, 2]
        self.sensor_data = data[:, 3:]
        self.data = data[:, 1:]


def all_recordings():
    return Recording(CO_PATH), Recording(METHANE_PATH)


# ================================================================
# Private
# ================================================================

@_memory.cache
def _read_gas_file(path):
    with open(path, 'r') as f:
        f.readline()
        data = np.fromstring(f.read(), dtype=np.float32, sep='\t')

    return data.reshape(-1, len(COLS))


def main():
    # import matplotlib.pyplot as plt
    # for r in all_recordings():
    #     print "recording {} has data of shape {}".format(r.name, r.data.shape)
    #     plt.plot(r.sampleTimes, r.sensor_data)
    #     plt.show()

    viz.plot_recordings(all_recordings(), interval_len=10000, savedir=FIG_SAVE_DIR)


if __name__ == '__main__':
    main()
