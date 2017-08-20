
import os
import numpy as np
import matplotlib.pyplot as plt
from joblib import Memory

import paths
from ..utils.files import listFilesInDir, ensure_dir_exists
from pamap_common import *  # noqa

memory = Memory('./')
join = os.path.join

# ================================================================
# consts

MISSING_DATA_VALUE = np.nan

OPTIONAL_DIR = join(paths.PAMAP2, 'Optional')
PROTOCOL_DIR = join(paths.PAMAP2, 'Protocol')
FIG_SAVE_DIR = join('figs', 'pamap2')
SAVE_DIR_LINE_GRAPH = join(FIG_SAVE_DIR, 'line')
SAVE_DIR_IMG = join(FIG_SAVE_DIR, 'img')

ACTIVITY_IDS_2_NAMES = {
    0: NAME_OTHER,
    1: NAME_LYING,
    2: NAME_SITTING,
    3: NAME_STANDING,
    4: NAME_WALK,
    5: NAME_RUN,
    6: NAME_CYCLE,
    7: NAME_NORDIC_WALK,
    9: NAME_WATCH_TV,
    10: NAME_COMPUTER_WORK,
    11: NAME_DRIVE,
    12: NAME_ASCEND_STAIRS,
    13: NAME_DESCEND_STAIRS,
    16: NAME_VACUUM,
    17: NAME_IRONING,
    18: NAME_FOLD_LAUNDRY,
    19: NAME_CLEANING,
    20: NAME_SOCCER,
    24: NAME_JUMP_ROPE
}

IMU_COL_NAMES = ['temp',
                 'accel16X', 'accel16Y', 'accel16Z',
                 'accelX', 'accelY', 'accelZ',
                 'gyroX', 'gyroY', 'gyroZ',
                 'magX', 'magY', 'magZ',
                 'null1', 'null2', 'null3', 'null4']
ALL_COL_NAMES = INITIAL_COL_NAMES[:]
ALL_COL_NAMES.extend([name + '_hand' for name in IMU_COL_NAMES])
ALL_COL_NAMES.extend([name + '_chest' for name in IMU_COL_NAMES])
ALL_COL_NAMES.extend([name + '_shoe' for name in IMU_COL_NAMES])


# ================================================================
# funcs

def getProtocolFilePaths():
    return listFilesInDir(PROTOCOL_DIR, endswith='.dat', absPaths=True)


def getOptionalFilePaths():
    return listFilesInDir(OPTIONAL_DIR, endswith='.dat', absPaths=True)


def getAllPamap2Recordings():
    for p in getProtocolFilePaths() + getOptionalFilePaths():
        yield Pamap2Recording(p)


# ================================================================
# recording class

class Pamap2Recording(Recording):

    def __init__(self, filePath):
        super(Pamap2Recording, self).__init__(
            filePath, MISSING_DATA_VALUE, ALL_COL_NAMES, ACTIVITY_IDS_2_NAMES)
        self.isOpt = OPTIONAL_DIR in filePath
        self.name = str(self)

    def __str__(self):
        s = "opt" if self.isOpt else "prot"
        return "subj%d_%s" % (self.subjId, s)


@memory.cache
def buildRecording(filePath):
    return Pamap2Recording(filePath)


# ================================================================
# main

if __name__ == '__main__':
    ensure_dir_exists(SAVE_DIR_LINE_GRAPH)
    ensure_dir_exists(SAVE_DIR_IMG)

    # r = buildRecording(PROTOCOL_DIR + '/subject101.dat')
    # plt.figure(figsize=(WIDTH_LINE_GRAPH, HEIGHT_LINE_GRAPH))
    # r.plot()
    # plt.figure(figsize=(4, 10))
    # r.imshow(znorm=True)
    # plt.show()
    # plt.savefig(FIG_SAVE_DIR + str(r))

    recs = getAllPamap2Recordings()
    for r in recs:
        print('plotting recording: ' + str(r))

        # plt.figure(figsize=(WIDTH_LINE_GRAPH, HEIGHT_LINE_GRAPH))
        # r.plot()
        # plt.savefig(join(SAVE_DIR_LINE_GRAPH, str(r)))

        plt.figure(figsize=(WIDTH_IMG, HEIGHT_IMG))
        r.imshow(znorm=True)
        plt.savefig(join(SAVE_DIR_IMG, str(r)))

    # plt.show()
