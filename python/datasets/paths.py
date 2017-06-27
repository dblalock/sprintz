#!/usr/env/python

import os

DATASETS_DIR = os.path.expanduser("~/Desktop/datasets/")

j = os.path.join     # abbreviation


def pathTo(subdir):
    return j(DATASETS_DIR, subdir)


MSRC_12 = pathTo(j('MSRC-12', 'origData'))
UCR = pathTo('ucr_data')
UWAVE = pathTo(j('uWave', 'extracted'))
PAMAP = pathTo('PAMAP_Dataset')
PAMAP2 = pathTo('PAMAP2_Dataset')
WARD = pathTo('WARD1.0')
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
