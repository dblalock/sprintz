#!/usr/bin/env python

import os
from joblib import Memory

import paths
from ..utils.files import listFilesInDir, ensure_dir_exists
from pamap_common import *  # noqa

memory = Memory('./')
join = os.path.join


# ================================================================
# consts

MISSING_DATA_VALUE = -1.0e6  # defined by data creators

# paths
INDOOR_DIR = join(paths.PAMAP, 'indoor')
OUTDOOR_DIR = join(paths.PAMAP, 'outdoor')
FIG_SAVE_DIR = join('figs', 'pamap')
SAVE_DIR_LINE_GRAPH = join(FIG_SAVE_DIR, 'line')
SAVE_DIR_IMG = join(FIG_SAVE_DIR, 'img')
SAVE_DIR_DELTA = join(FIG_SAVE_DIR, 'delta')

# activity names
ACTIVITY_IDS_2_NAMES = {
    0: NAME_OTHER,
    1: NAME_LYING,
    2: NAME_SITTING,
    3: NAME_STANDING,
    10: NAME_SLOW_WALK,
    11: NAME_WALK,
    12: NAME_NORDIC_WALK,
    13: NAME_RUN,
    14: NAME_ASCEND_STAIRS,
    15: NAME_DESCEND_STAIRS,
    16: NAME_CYCLE,
    20: NAME_IRONING,
    21: NAME_VACUUM,
    22: NAME_JUMP_ROPE,
    23: NAME_SOCCER
}

# column names
IMU_COL_NAMES = ['temp',
                 'accelX', 'accelY', 'accelZ',
                 'gyroX', 'gyroY', 'gyroZ',
                 'magX', 'magY', 'magZ',
                 'null1', 'null2', 'null3', 'null4']
ALL_COL_NAMES = INITIAL_COL_NAMES[:]
ALL_COL_NAMES.extend([name + '_hand' for name in IMU_COL_NAMES])
ALL_COL_NAMES.extend([name + '_chest' for name in IMU_COL_NAMES])
ALL_COL_NAMES.extend([name + '_shoe' for name in IMU_COL_NAMES])


# ================================================================
# utility funcs

def getIndoorFilePaths():
    return listFilesInDir(INDOOR_DIR, endswith='.dat', absPaths=True)


def getOutdoorFilePaths():
    return listFilesInDir(OUTDOOR_DIR, endswith='.dat', absPaths=True)

# def dfFromFileAtPath(path):
#     # read in the data file and pull out the
#     # columns with valid data (and also replace
#     # their missing data marker with nan
#     data = np.genfromtxt(path)
#     data[data == MISSING_DATA_VALUE] = np.nan
#     df = pd.DataFrame(data=data, columns=ALL_COL_NAMES)
#     return df.filter(COL_NAMES)


@memory.cache
def all_recordings():
    recs = []
    for p in getIndoorFilePaths() + getOutdoorFilePaths():
        print "reading pamap file: ", p
        # yield PamapRecording(p)
        recs.append(PamapRecording(p))
    return recs


# ================================================================
# recording class

class PamapRecording(Recording):

    def __init__(self, filePath):
        super(PamapRecording, self).__init__(
            filePath, MISSING_DATA_VALUE, ALL_COL_NAMES, ACTIVITY_IDS_2_NAMES)
        self.isIndoor = INDOOR_DIR in filePath
        self.name = str(self)

    def __str__(self):
        s = "in" if self.isIndoor else "out"
        return "subj%d_%s" % (self.subjId, s)


@memory.cache
def buildRecording(filePath):
    return PamapRecording(filePath)


# ================================================================
# main

def main():

    ensure_dir_exists(SAVE_DIR_LINE_GRAPH)
    ensure_dir_exists(SAVE_DIR_IMG)
    ensure_dir_exists(SAVE_DIR_DELTA)

    # r = buildRecording(INDOOR_DIR + '/subject1.dat')
    # plt.figure(figsize=(WIDTH_LINE_GRAPH, HEIGHT_LINE_GRAPH))
    # r.plot()
    # plt.figure(figsize=(WIDTH_IMG, HEIGHT_IMG))
    # r.imshow(znorm=True)
    # plt.show()
    # plt.savefig(SAVE_DIR_LINE_GRAPH + str(r))

    recs = all_recordings()

    print "recording 0 shape: ", recs[0].data.shape

    # for i, r in enumerate(recs):
    #     print('plotting recording: ' + str(r))

    #     # plt.figure(figsize=(WIDTH_LINE_GRAPH, HEIGHT_LINE_GRAPH))
    #     # r.plot()
    #     # # plt.savefig(join(SAVE_DIR_LINE_GRAPH, str(r)))

    #     plt.figure(figsize=(WIDTH_IMG, HEIGHT_IMG))
    #     r.imshow(znorm=True)
    #     plt.savefig(join(SAVE_DIR_IMG, str(r)))

    #     if i > 4:
    #         break

    # plt.show()

    # viz.plot_recordings(recs, interval_len=10000, savedir=SAVE_DIR_DELTA)


if __name__ == '__main__':
    main()
