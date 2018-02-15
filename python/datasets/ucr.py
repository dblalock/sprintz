#!/usr/bin/env/python

import os
import numpy as np
from joblib import Memory

import paths
from utils import concatedTsList


_memory = Memory('./')

UCR_DATASETS_DIR = paths.UCR
ORIG_UCR_DATASETS_DIR = paths.UCR_ORIG

SHORT_UCR_DATASETS = [  # as determined by instance length, not number thereof
    "ItalyPowerDemand",
    "synthetic_control",
    "SonyAIBORobotSurfaceII",
    "SonyAIBORobotSurface",
    "TwoLeadECG",
    "MoteStrain",
    "ECG200",
    "MedicalImages",
    "CBF",
    "SwedishLeaf",
    "Two_Patterns",
    "FaceAll",
    "FacesUCR",
    "ECGFiveDays",
    "Gun_Point",
    "wafer",
    "ChlorineConcentration",
    "Adiac",
    "50words",
    "WordsSynonyms"
]


# ================================================================
# Public
# ================================================================

def allUCRDatasets():
    for dataDir in sorted(allUCRDatasetDirs()):
        yield UCRDataset(dataDir, sep=',')


def origUCRDatasets():
    for dataDir in sorted(origUCRDatasetDirs()):
        yield UCRDataset(dataDir)


def smallUCRDatasets():
    for dataDir in sorted(smallUCRDatasetDirs()):
        yield UCRDataset(dataDir)


class UCRDataset(object):

    def __init__(self, datasetDir, sep=None):
        self.Xtrain, self.Ytrain = readUCRTrainData(datasetDir, sep=sep)
        self.Xtest, self.Ytest = readUCRTestData(datasetDir, sep=sep)
        self.name = nameFromDir(datasetDir)

        self.X = np.r_[self.Xtrain, self.Xtest]
        self.Y = np.r_[self.Ytrain, self.Ytest]


def readUCRDataset(datasetDir, useTrain=True, useTest=True):
    dataset = UCRDataset(datasetDir)
    if (not useTrain) and useTest:
        X, Y = dataset.Xtest, dataset.Ytest
    elif useTrain and (not useTest):
        X, Y = dataset.Xtrain, dataset.Ytrain
    elif useTrain and useTest:
        X, Y = dataset.X, dataset.Y
    else:
        raise ValueError("Must use training or testing data!")

    return X, Y


def labeledTsListFromDataset(datasetDir, useTrain=True, useTest=True,
                             instancesPerTs=10, minPaddingFraction=1.25,
                             **paddingKwargs):

    if not os.path.exists(datasetDir):
        datasetDir = dirFromName(datasetDir)
    if not os.path.exists(datasetDir):
        raise ValueError("Couldn't load unrecognized dataset {}".format(
            datasetDir))

    X, Y = readUCRDataset(datasetDir, useTrain=useTrain, useTest=useTrain)
    datasetName = os.path.basename(datasetDir)
    tsList = concatedTsList(X, Y, instancesPerTs=instancesPerTs,
                            datasetName=datasetName, std=.25,
                            minPaddingFractionOfLength=minPaddingFraction,
                            **paddingKwargs)

    np.random.shuffle(tsList)
    return tsList


def allUCRDatasetDirs():
    return _ucr_datasets_in_dir(UCR_DATASETS_DIR)


def origUCRDatasetDirs():
    return _ucr_datasets_in_dir(ORIG_UCR_DATASETS_DIR)


def smallUCRDatasetDirs():
    return [dirFromName(name) for name in SHORT_UCR_DATASETS]


# ================================================================
# Private
# ================================================================

def _ucr_datasets_in_dir(dirpath):
    datasetsPath = os.path.expanduser(dirpath)
    files = os.listdir(datasetsPath)
    for i in range(len(files)):
        files[i] = os.path.join(datasetsPath, files[i])
    dirs = filter(os.path.isdir, files)
    return dirs


@_memory.cache
def _readtxt(path, sep=None):
    return np.genfromtxt(path, delimiter=sep)


def readDataFile(path, sep=None, mean_norm=True):
    D = _readtxt(path, sep=sep)
    labels = D[:, 0].astype(np.int)
    X = D[:, 1:]
    if mean_norm:
        X -= np.mean(X, axis=1, keepdims=True)
    return (X, labels)


def nameFromDir(datasetDir):
    return os.path.basename(datasetDir)


def dirFromName(datasetName):
    return os.path.join(paths.UCR, datasetName)


def readUCRDataInDir(datasetDir, train, sep=None):
    datasetName = nameFromDir(datasetDir)
    if train:
        fileName = datasetName + "_TRAIN"
    else:
        fileName = datasetName + "_TEST"
    filePath = os.path.join(datasetDir, fileName)
    return readDataFile(filePath, sep=sep)


def readUCRTrainData(datasetDir, sep=None):
    return readUCRDataInDir(datasetDir, train=True, sep=sep)


def readUCRTestData(datasetDir, sep=None):
    return readUCRDataInDir(datasetDir, train=False, sep=sep)


# combines train and test data
def readAllUCRData(ucrDatasetDir):
    X_train, Y_train = readUCRTrainData(ucrDatasetDir)
    X_test, Y_test = readUCRTestData(ucrDatasetDir)
    X = np.r_[X_train, X_test]
    Y = np.r_[Y_train, Y_test]
    return X, Y


# ================================================================ Main

def main():
    from matplotlib import pyplot as plt

    dsets = allUCRDatasets()
    for dset in dsets:
        print "dset name:", dset.name
    return

    printTable = False
    if printTable:
        # print out a table of basic stats for each dataset to verify
        # that everything is working
        nameLen = 22
        print("%s\tTrain\tTest\tLength\tClasses" % (" " * nameLen))
        for i, datasetDir in enumerate(allUCRDatasetDirs()):
            Xtrain, _ = readUCRTrainData(datasetDir)
            Xtest, Ytest = readUCRTestData(datasetDir)
            print('%22s:\t%d\t%d\t%d\t%d' % (nameFromDir(datasetDir),
                  Xtrain.shape[0], Xtest.shape[0], Xtrain.shape[1],
                  len(np.unique(Ytest))))

    plotTs = True
    if plotTs:
        saveDir = 'figs/ucr/groups/'
        howManyPerTs = 5  # number of ts to plot from each dataset
        howManyPerDataset = 50
        np.random.seed(123)  # so we always create and select the same groups

        # for dataDir in allUCRDatasetDirs():
        for dataDir in smallUCRDatasetDirs():
            tsFromDataset = labeledTsListFromDataset(
                dataDir, instancesPerTs=howManyPerTs)[:howManyPerDataset]
            saveSubdir = os.path.join(saveDir, nameFromDir(dataDir))
            for ts in tsFromDataset:
                print ts.name, ts.data.shape
                ts.plot(saveSubdir, staggerHeights=False)

    plotSubseqs = False
    if plotSubseqs:
        saveDir = 'figs/ucr/concat/'
        howManyPerDataset = 5  # number of ts to plot from each dataset

        # for dataDir in allUCRDatasetDirs():
        for dataDir in smallUCRDatasetDirs():
            tsFromDataset = labeledTsListFromDataset(
                dataDir, instancesPerTs=howManyPerDataset)
            saveSubdir = os.path.join(saveDir, nameFromDir(dataDir))
            for ts in tsFromDataset:
                print ts.name, ts.data.shape
                windowLen = int(len(ts.data) / len(ts.labels))
                vidLen = len(ts.data) + 1
                ts.plotSubseqs(saveSubdir, windowLen=windowLen, step=vidLen,
                               createSubdir=False)
                plt.close()

        # howMany = 2
        # tsList = []
        # tsList += labeledTsListFromDataset('Trace')[:howMany]
        # tsList += labeledTsListFromDataset('Fish')[:howMany]
        # tsList += labeledTsListFromDataset('Coffee')[:howMany]
        # tsList += labeledTsListFromDataset('CBF')[:howMany]
        # tsList += labeledTsListFromDataset('ECG200')[:howMany]
        # for ts in tsList:
        #     print ts.name, ts.data.shape
        #     ts.plotSubseqs(saveDir)


if __name__ == '__main__':
    main()
