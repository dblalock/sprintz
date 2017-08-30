
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Memory

from ..utils.files import basename
from ..utils.arrays import downsampleMat, zNormalizeCols, zeroOneScaleMat
from ..utils import sequence as seq

memory = Memory('./', verbose=1)

# ================================================================
# numeric consts

# clipping / value manipulation
MINVAL = -150.
MAXVAL = 175.

WIDTH_LINE_GRAPH = 13
HEIGHT_LINE_GRAPH = 6
WIDTH_IMG = 4
HEIGHT_IMG = 10

# ================================================================
# Column names in data
TIMESTAMP_COL_NAME = 'time'
LABEL_COL_NAME = 'activity_id'
INITIAL_COL_NAMES = [TIMESTAMP_COL_NAME, LABEL_COL_NAME, 'heartRate']

# ================================================================
# Activity IDs

OTHER_ACTIVITY_ID = 0

# ================================================================
# Activity names

# shared
NAME_OTHER = 'NA'
NAME_LYING = 'lying'
NAME_SITTING = 'sitting'
NAME_STANDING = 'standing'
NAME_VACUUM = 'vacuum'
NAME_WALK = 'walk'
NAME_NORDIC_WALK = 'Nordic walk'
NAME_ASCEND_STAIRS = 'ascend stairs'
NAME_DESCEND_STAIRS = 'descend stairs'
NAME_RUN = 'run'
NAME_SOCCER = 'soccer'
NAME_JUMP_ROPE = 'jump rope'
NAME_CYCLE = 'cycle'
NAME_IRONING = 'iron'

# pamap only
NAME_SLOW_WALK = 'slow walk'

# pamap2 only
NAME_WATCH_TV = 'watch TV'
NAME_COMPUTER_WORK = 'computer work'
NAME_DRIVE = 'drive'
NAME_FOLD_LAUNDRY = 'folding laundry'
NAME_CLEANING = 'cleaning'


# ================================================================
# funcs

def removeNullCols(colNames):
    return filter(lambda name: 'null' not in name, colNames)


# -------------------------------
# data parsing funcs
# -------------------------------

def parseDataFileName(f):
    name = basename(f, noexts=True)
    subjId = int(name[-1])
    return subjId


@memory.cache
def _readtxt(path):
    return np.genfromtxt(path)


@memory.cache
def dfFromFileAtPath(path, missingDataVal, allColNames, keepColNames):
    # read in the data file and pull out the
    # columns with valid data (and also replace
    # their missing data marker with nan
    data = _readtxt(path)
    data[data == missingDataVal] = np.nan
    df = pd.DataFrame(data=data, columns=allColNames)
    return df.filter(keepColNames)


def findActivityBoundaries(df, labelColName=LABEL_COL_NAME):
    labelCol = df[labelColName]
    boundaries = seq.rangesOfConstantValue(labelCol)
    labels = [labelCol[row[0]] for row in boundaries]
    assert(len(labels) == len(boundaries))
    return boundaries, labels


# -------------------------------
# plotting funcs
# -------------------------------

def plotVertLine(x, ymin, ymax):
    plt.plot([x, x], [ymin, ymax], color='k', linestyle='--', linewidth=1)


def imshowData(data, znorm=False):
    if (znorm):
        data = zNormalizeCols(data)
    data = zeroOneScaleMat(data)
    plt.imshow(data, aspect='auto')
    plt.colorbar()


def plotRecording(sampleTimes, data, boundaries, labelStrings,
                  minVal=MINVAL, maxVal=MAXVAL):
    maxTimestamp = sampleTimes[-1]
    plt.gca().set_autoscale_on(False)
    plt.plot(sampleTimes, data)
    plt.xlabel("Times (s)")
    for i, row in enumerate(boundaries):
        # plot line
        idx = row[0]
        timestamp = sampleTimes[idx]
        plotVertLine(timestamp, minVal, maxVal)

        # write label
        x = timestamp / maxTimestamp
        y = .05 + (.8 * (i % 2)) + (.025 * (i % 4))     # stagger heights
        name = labelStrings[i]
        plt.gca().annotate(name, xy=(x, y), xycoords='axes fraction')

    plt.xlim([np.min(sampleTimes), np.max(sampleTimes)])
    plt.ylim([minVal, maxVal])


# ================================================================
# recording class

# @memory.cache
# def _array_from_df(df, columns):
    # return df.as_matrix(columns=columns)

class Recording(object):

    def __init__(self, filePath, missingDataVal, colNames, ids2labelStrs):
        dataColNames = colNames[2:]
        usedColNames = removeNullCols(colNames)

        self.subjId = parseDataFileName(filePath)
        self.df = dfFromFileAtPath(filePath, missingDataVal,
                                   colNames, usedColNames)
        self.sampleTimes = self.df.as_matrix([TIMESTAMP_COL_NAME])
        self.data = self.df.as_matrix(columns=dataColNames)
        np.clip(self.data, MINVAL, MAXVAL, out=self.data)
        self.boundaries, self.labels = findActivityBoundaries(self.df)
        self.labelStrs = [ids2labelStrs[label] for label in self.labels]

    def plot(self):
        plotRecording(self.sampleTimes, self.data,
                      self.boundaries, self.labelStrs)

    def imshow(self, znorm=False):
        data = self.data
        data[np.isnan(data)] = MINVAL  # znorming breaks everything without this
        # downsample by k cuz otherwise a whole bunch of rows get averaged
        # together in the plot and the whole thing is just ~.5
        data = downsampleMat(data, rowsBy=4)
        imshowData(data, znorm)
