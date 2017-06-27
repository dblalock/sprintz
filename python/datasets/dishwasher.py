#!/usr/bin/env/python

import os
import numpy as np
import pandas as pd

import paths
from utils import saveAnimation, animateSubseqs, dataNearAnnotations
from utils import generateVideos, sectionsOfDataNearAnnotationsImpure

VIDS_DIR = 'vids'


# ------------------------------------------------ Public funcs

def getLabeledTsList(instancesPerTs=10, shuffle=False, padLen=25,
                     keepLabels=['Z', 'Z2'], includeZC=True, padJitter=175,
                     addNoise=True):

    r = Recording(padLen=padLen, includeZC=includeZC, addNoise=addNoise)

    if includeZC:
        # keepLabels = ['Z3'] # superset of Z2
        keepLabels = ['Z', 'Z3', 'Z2', 'ZC']  # superset of Z2

    tsList = sectionsOfDataNearAnnotationsImpure(
        r.data, r.rangeStartIdxs,
        r.rangeEndIdxs, r.labels, instancesPerTs=instancesPerTs, shuffle=shuffle,
        padLen=padLen, maxPadJitter=padJitter, keepLabels=keepLabels,
        datasetName="dishwasherGroups")

    return tsList


# ------------------------------------------------ File IO

def readFile(path):
    return np.loadtxt(path, delimiter=',')


def readData():
    return readFile(paths.DISHWASHER)[:, 1:]  # first col is just timestamp
    # return readFile(paths.DISHWASHER_SHORT)
    # return readFile(paths.DISHWASHER_20K)


def readAnnotations():
    try:
        annotations = pd.read_csv(paths.DISHWASHER_LABELS,
                                  delim_whitespace=True, header=None)
    except IOError:  # happens if not calling this from project root
        # if calling if from one level above project root (for standalone
        # code), this fixes it
        annotations = pd.read_csv(paths.DISHWASHER_LABELS_ALT,
                                  delim_whitespace=True, header=None)
    annotations.columns = ['start', 'end', 'label']
    return annotations


# ------------------------------------------------ Old funcs

def generateVids():
    """main function used to generate videos for dataset annotation; see function
    in utils for a much cleaner version. This function should not be used."""

    # assumes header line has been removed; on unix, this can be done via:
    #     $ cat DWE.csv | tail -n +2 > dishwasher_nohead.csv
    #
    # Note that you will of course have to have the csv saved in the current
    # directory for this (or the rest of this script) to work
    appliance = 'dishwasher'
    # appliance = 'washer'
    # appliance = 'dryer'
    path = appliance + '_nohead.csv'
    ar = readFile(path)
    # ar = ar[:(20*1000)]
    print("{0} -> {1} array".format(path, ar.shape))

    # # find rising edges and save where they are to help us annotate
    # realPower = ar[:, 6]
    # diffs = realPower[1:] - realPower[:-1]
    # # idxs = np.where((realPower[1:] > 500) * (diffs > 100.))[0]
    # idxs = np.where(diffs > 50.)[0]
    # np.savetxt('idxs.txt', idxs, fmt="%d")
    # return

    # # plot a few particular indices we accidentally missed (bug is
    # # fixed now, but I don't want to recreate all the videos)
    # missedIdxs = [354950, 379950, 559950, 634950, 1014950]
    # for idx in missedIdxs:
    #     xMin = idx - 100
    #     xMax = idx + 200
    #     xVals = np.arange(xMin, xMax, dtype=np.int)
    #     offset = int((xMax // 1e5) * 1e5) # floor to nearest 100k
    #     plt.figure(figsize=(8,6))
    #     plt.title("{0}, {1}-{2}".format(appliance.title(), xMin, xMax))
    #     plt.plot(xVals - offset, ar[xMin:xMax])
    #     plt.xlim([xMin - offset, xMax - offset])
    #     plt.ylim([0,1000])
    # plt.show()
    # return

    step = 5*1000
    windowLen = 300
    epochSz = 100*1000  # 100k per epoch--mostly so xlabels stay legible
    for epochNum, epochStartIdx in enumerate(range(0, len(ar), epochSz)):
        epochEndIdx = epochStartIdx + epochSz
        epochData = ar[epochStartIdx:epochEndIdx]

        subdir = "{0}k-{1}k".format(epochStartIdx / 1000, epochEndIdx / 1000)
        saveDir = VIDS_DIR + '/' + appliance + '/' + subdir + '/'
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

        n = len(epochData)
        for startIdx in range(0, n, step):
            endIdx = min(startIdx + step, n)
            absoluteStartIdx = epochStartIdx + startIdx
            absoluteEndIdx = epochStartIdx + endIdx
            data = epochData[startIdx:endIdx]

            figName = appliance + "_{0}-{1}".format(absoluteStartIdx, absoluteEndIdx-1)
            figPath = os.path.join(saveDir, figName + '.mp4')

            anim = animateSubseqs(data, windowLen, figsize=(8, 6),
                                  dataName=appliance.title(), ylimits=[0, 1000],
                                  idxOffsetTitle=absoluteStartIdx,
                                  idxOffsetXLabel=startIdx)
            saveAnimation(anim, figPath, fps=25)


# ------------------------------------------------ Data Structures

def addZC(startIdxs, endIdxs, labels, alsoZ3=True):

    newStartIdxs = []
    newEndIdxs = []
    newLabels = []

    for i in range(len(startIdxs) - 1):
        # Z, C -> Z, ZC, Z3
        # Z, Z2 -> Z
        # Z2, C -> Z2, ZC, Z3
        # Z, * -> Z, Z3
        # Z2, * -> Z2, Z3

        newStartIdxs.append(startIdxs[i])
        newEndIdxs.append(endIdxs[i])
        newLabels.append(labels[i])

        isZ = labels[i] == 'Z'
        isZ2 = labels[i] == 'Z2'
        if not (isZ2 or isZ):
            continue

        nextIsZ2 = labels[i+1] == 'Z2'
        if nextIsZ2:
            continue

        z2start = startIdxs[i]
        cEnd = endIdxs[i+1]  # end of (possible) C after the Z2
        followedByC = (labels[i+1] == 'C') and (cEnd - endIdxs[i] < 150)
        if followedByC:
                newStartIdxs.append(z2start)
                newEndIdxs.append(cEnd)
                newLabels.append('ZC')
        else:
            cEnd = endIdxs[i]  # just the end of the Z2

        if alsoZ3:
            newStartIdxs.append(z2start)
            newEndIdxs.append(cEnd)
            newLabels.append('Z3')  # Z2 without a C or ZC

    newStartIdxs = np.array(newStartIdxs, dtype=np.int)
    newEndIdxs = np.array(newEndIdxs, dtype=np.int)
    newLabels = np.array(newLabels, dtype=np.object)

    return newStartIdxs, newEndIdxs, newLabels


class Recording(object):

    def __init__(self, shortened=False, padLen=100, just2=False, just3=False,
                 includeZC=True, addNoise=True):

        self.data = readData()

        if addNoise:
            self.data += np.random.randn(self.data.shape[0], self.data.shape[1])

        annos = readAnnotations()
        # annos = annos[:22] # for dishwasher_20k
        self.rangeStartIdxs = np.array(annos['start'], dtype=np.int)
        self.rangeEndIdxs = np.array(annos['end'], dtype=np.int)
        self.labels = np.array(annos['label'], dtype=np.str)

        self.shortened = shortened or just2 or just3
        self.just2 = just2  # only 2 examples
        self.just3 = just3  # only 3 examples

        if padLen >= 0:
            self.padLen = padLen
        else:  # negative padLen -> set automatically
            padLen = np.mean(self.rangeEndIdxs - self.rangeStartIdxs) / 2.

        if self.shortened:
            self.data, self.rangeStartIdxs, self.rangeEndIdxs, = \
                dataNearAnnotations(self.data, self.rangeStartIdxs,
                                    self.rangeEndIdxs, self.padLen)

        if self.just2 or self.just3:
            whereZ2 = np.where(self.labels == 'Z2')[0]
            if self.just2:
                keepIdxs = whereZ2[:2]
                self.data = self.data[:650]  # after end of 2nd instance
                # self.labels = np.zeros(2)
            elif self.just3:
                keepIdxs = whereZ2[:3]
                self.data = self.data[:950]  # after end of 3rd instance
                # self.labels = np.zeros(3)

            self.rangeStartIdxs = self.rangeStartIdxs[keepIdxs]
            self.rangeEndIdxs = self.rangeEndIdxs[keepIdxs]
            self.labels = self.labels[keepIdxs]

        if includeZC:  # also have Z2 followed by C as a pattern, since common
            self.rangeStartIdxs, self.rangeEndIdxs, self.labels = addZC(
                self.rangeStartIdxs, self.rangeEndIdxs, self.labels)

        print "Dishwasher recording: data shape = ", self.data.shape
        # print "Dishwasher annotations:"
        # annotations = np.c_[self.rangeStartIdxs, self.rangeEndIdxs, self.labels]
        # for anno in annotations:
        #     if 50 <= int(anno[0]) // 1000 < 60: print anno

    def animate(self):
        dataName = 'dishwasher'
        if self.shortened:
            dataName += '-short'
            dataName += '-pad{}'.format(self.padLen)
        if self.just2:
            dataName += '-just2'
        elif self.just3:
            dataName += '-just3'

        # generateVideos(self.data, dataName="dishwasher", saveInDir="figs")
        # generateVideos(self.data, dataName="dishwasher-short", saveInDir="figs")
        generateVideos(self.data,
                       dataName=dataName,
                       saveInDir="figs",
                       rangeStartIdxs=self.rangeStartIdxs,
                       rangeEndIdxs=self.rangeEndIdxs,
                       rangeLabels=self.labels,
                       ylimits=[0, 1000])


# ------------------------------------------------ Main

if __name__ == '__main__':
    from doctest import testmod
    testmod()

    # r = Recording(shortened=True)
    # r = Recording(shortened=False)
    # r = Recording(just2=True)
    # r = Recording(just3=True)
    # r.animate()

    # print np.median(r.data, axis=0)
    # print np.mean(r.data, axis=0)

    # plt.plot(r.data)
    # plt.show()

    #
    # plot groups with 5 instances
    #
    # saveDir = 'figs/dishwasher_groups/'
    # np.random.seed(123)
    # tsList = getLabeledTsList(shuffle=True, includeZC=True, instancesPerTs=5)
    # # note that grouping into 5s with this seed yields 2 pretty good visual
    # # examples as the last group--one is missing part of the X
    # for ts in tsList:
    #     # ts.plot(saveDir=saveDir, staggerHeights=False)
    #     ts.plot(saveDir=saveDir)

    #
    # plot groups with 3 instances
    #
    saveDir = 'figs/dishwasher_groups3/'
    np.random.seed(123)
    tsList = getLabeledTsList(shuffle=True, includeZC=True, instancesPerTs=3)
    # note that grouping into 5s with this seed yields 2 pretty good visual
    # examples as the last group--one is missing part of the X
    for ts in tsList:
        # ts.plot(saveDir=saveDir, staggerHeights=False)
        ts.plot(saveDir=saveDir)


    # annos = readAnnotations()
    # whereZ2 = np.where(annos['label'] == 'Z2')[0]
    # z2starts = np.array(annos['start'][whereZ2])
    # z2ends = np.array(annos['end'][whereZ2])
    # # print whereZ2, z2starts, z2ends
    # lengths = z2ends - z2starts
    # print lengths
    # minIdx = np.argmin(lengths)
    # print minIdx, z2starts[minIdx]


