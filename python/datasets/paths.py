#!/usr/env/python

import os

DATASETS_DIR = os.path.expanduser("~/Desktop/datasets/")

j = os.path.join     # abbreviation


def pathTo(subdir):
    return j(DATASETS_DIR, subdir)


# straightforward datasets
MSRC_12 = pathTo(j('MSRC-12', 'origData'))
UCR_ORIG = pathTo('ucr_data')
UCR = pathTo('UCR_TS_Archive_2015')
UWAVE = pathTo(j('uWave', 'extracted'))
PAMAP = pathTo('PAMAP_Dataset')
PAMAP2 = pathTo('PAMAP2_Dataset')
WARD = pathTo('WARD1.0')
UCI_GAS = pathTo('uci-gas-sensor')

# ampds2
AMPD2_POWER = pathTo(j('ampds2', 'electric'))
# AMPD2_POWER = pathTo(j('ampds2', 'electric', 'debug_all_power.csv')) # TODO
AMPD2_GAS = pathTo(j('ampds2', 'natural_gas'))
AMPD2_WEATHER = pathTo(j('ampds2', 'climate_weather'))
AMPD2_WATER = pathTo(j('ampds2', 'water'))

# ampds
DISHWASHER = pathTo(j('AMPds', 'dishwasher_nohead.csv'))
DISHWASHER_SHORT = pathTo(j('AMPds', 'dishwasher_nohead_short.csv'))
DISHWASHER_20K = pathTo(j('AMPds', 'dishwasher_nohead_20k.csv'))
DISHWASHER_LABELS = 'python/datasets/dishwasher-labels.txt'  # in project dir
DISHWASHER_LABELS_ALT = 'ts/python/datasets/dishwasher-labels.txt'  # proj dir


# TIDIGITS; the executable can be compiled from the source code here:
# https://www.ldc.upenn.edu/language-resources/tools/sphere-conversion-tools
# We just happen to have placed it in our tidigits subdirectory for
# convenience; it is not included with TIDIGITS.
TIDIGITS = pathTo(j('tidigits', 'data'))
SPH2PIPE_EXECUTABLE = pathTo(j('tidigits', 'sph2pipe_v2.5', 'sph2pipe'))
