#!/usr/env/python

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Memory

import dishwasher as dw
import msrc
import synthetic as synth
import tidigits as ti
import ucr
from utils import LabeledTimeSeries

_memory = Memory('.', verbose=1)

# TODO specify these elsewhere
SYNTHETIC_DATA_NOISE = .05
DEFAULT_RAND_WALK_LENGTH = 1000


# ================================ Dataset Names

DISHWASHER = 'dishwasher_groups'
TIDIGITS = 'tidigits_grouped_mfccs'
MSRC = 'msrc'
UCR = 'ucr_short'
TRIANGLES = 'triangles'
RECTS = 'rects'
SINES = 'sines'
SHAPES = 'shapes'
RANDWALK = 'randwalk'


# ================================ UCR Time Series

def allUCRDatasets():
    return ucr.allUCRDatasets()


def origUCRDatasets():
    return ucr.origUCRDatasets()


@_memory.cache
def smallUCRDatasets():
    return filter(lambda d: d.Xtrain.size < 50e3, allUCRDatasets())


@_memory.cache
def tinyUCRDatasets():
    return filter(lambda d: d.Xtrain.size < 10e3, smallUCRDatasets())


# ================================ DataLoader

@_memory.cache
def loadDataset(datasetName, seed=None, whichExamples=None, instancesPerTs=5,
                minNumInstances=2, maxNumInstances=None, cropDataLength=None):
    if seed:
        print("loadDataset(): seeding RNG with value {}".format(seed))
        synth.seedRng(seed)

    hasWhichExamples = whichExamples is not None and len(whichExamples)

    # ------------------------ synthetic datasets

    func = None
    if datasetName == TRIANGLES:
        func = synth.trianglesMotif
    elif datasetName == RECTS:
        func = synth.rectsMotif
    elif datasetName == SINES:
        func = synth.sinesMotif
    elif datasetName == SHAPES:
        func = synth.multiShapesMotif

    if func:
        if whichExamples is None or not len(whichExamples):
            whichExamples = [0]
        tsList = []
        for i in range(len(whichExamples)):
            (data, start1, start2), m = func(
                returnStartIdxs=True, noise=SYNTHETIC_DATA_NOISE,
                backgroundNoise=SYNTHETIC_DATA_NOISE)
            startIdxs = np.array([start1, start2], dtype=np.int)
            endIdxs = startIdxs + m

            name = '{}_{}'.format(datasetName, i)
            ts = LabeledTimeSeries(data, startIdxs, endIdxs, name=name)
            tsList.append(ts)
        return tsList

    if datasetName == RANDWALK:
        if whichExamples is None or not len(whichExamples):
            whichExamples = [0]
        tsList = []
        for i in range(len(whichExamples)):
            length = DEFAULT_RAND_WALK_LENGTH
            if cropDataLength:
                length = cropDataLength
            data = synth.randwalk((length, 1))
            name = 'randwalk-{}_{}'.format(length, i)
            ts = LabeledTimeSeries(data, [0], [1], name=name)
            tsList.append(ts)
        return tsList

    # ------------------------ real datasets

    # msrc
    if datasetName == 'msrc':
        recordings = list(msrc.getRecordings(idxs=whichExamples))
        tsList = []
        for r in recordings:
            startIdxs = r.gestureIdxs
            endIdxs = startIdxs
            name = 'msrc' + str(r.id) + r.gestureLabel
            ts = LabeledTimeSeries(r.data, startIdxs, endIdxs,
                                   name=name, id=r.id)
            tsList.append(ts)

    # dishwasher
    elif datasetName == 'dishwasher':
        r = dw.Recording(shortened=False)
        name = "Dishwasher"
        tsList = [LabeledTimeSeries(r.data, r.rangeStartIdxs, r.rangeEndIdxs,
                                    labels=r.labels, name=name)]
    elif datasetName == 'dishwasher_short':
        r = dw.Recording(shortened=True)
        name = "DishwasherShort"
        tsList = [LabeledTimeSeries(r.data, r.rangeStartIdxs, r.rangeEndIdxs,
                                    labels=r.labels, name=name)]
    elif datasetName == 'dishwasher_2':
        r = dw.Recording(just2=True)
        name = "Dishwasher2"
        tsList = [LabeledTimeSeries(r.data, r.rangeStartIdxs, r.rangeEndIdxs,
                                    labels=r.labels, name=name)]
    elif datasetName == 'dishwasher_3':
        r = dw.Recording(just3=True)
        name = "Dishwasher3"
        tsList = [LabeledTimeSeries(r.data, r.rangeStartIdxs, r.rangeEndIdxs,
                                    labels=r.labels, name=name)]
    elif datasetName == 'dishwasher_groups' or datasetName == 'dishwasher_pairs':
        tsList = dw.getLabeledTsList(instancesPerTs=instancesPerTs)
        if hasWhichExamples:
            tsList = [tsList[i] for i in whichExamples]

    # tidigits
    elif 'tidigits' in datasetName:
        if 'grouped' in datasetName:
            singleDigitsOnly = 'single' in datasetName
            recordings = ti.getConcatenatedRecordingsForDigits(
                singleDigitsOnly=singleDigitsOnly, instancesPerTs=instancesPerTs)
        else:
            recordings = ti.getAllRecordings()

        if hasWhichExamples:
            recordings = [recordings[i] for i in whichExamples]

        tsList = []

        for r in recordings:
            useMFCCs = 'mfcc' in datasetName
            if not useMFCCs:
                print("WARNING: TIDIGITS start and end indices only valid for"
                      "mfccs (because of a hack--see "
                      "getConcatenatedRecordingsForDigits)")

            lbls = r.digits
            representation = 'mfccs' if useMFCCs else 'raw'
            startIdxs, endIdxs = r.startEndIdxs(whichRepresentation=representation)
            if startIdxs is None:
                startIdxs = np.arange(len(lbls))  # dummy start idxs
            if endIdxs is None:
                endIdxs = startIdxs
            name = r.name

            # figure out the form in which we're loading it
            if 'raw' in datasetName:
                data = r.data
            elif 'mfcc' in datasetName:
                data = r.mfccs
            elif 'log_filterbank' in datasetName:
                data = r.logfbank
            elif 'filterbank' in datasetName:  # XXX must be after previous
                data = r.fbank
            else:
                raise ValueError("Unrecognized tidigits dataset: {}".format(
                    datasetName))

            ts = LabeledTimeSeries(data, startIdxs, endIdxs,
                                   labels=lbls, name=name)
            tsList.append(ts)

    # ucr
    elif datasetName[:4] == 'ucr_':
        datasetName = datasetName[4:]
        ucrAll = datasetName == 'all'
        ucrShort = datasetName == 'short'
        ucrPairs = datasetName == 'pairs'
        if ucrAll or ucrShort or ucrPairs:  # ucr_all | ucr_short | ucr_pairs
            if ucrAll:
                dirs = ucr.allUCRDatasetDirs()
            else:
                dirs = ucr.smallUCRDatasetDirs()

            if ucrPairs:
                instancesPerTs = 2

            tsList = []
            for dataDir in dirs:
                tsFromDataset = ucr.labeledTsListFromDataset(
                    dataDir, instancesPerTs=instancesPerTs)
                if hasWhichExamples:
                    tsFromDataset = [tsFromDataset[i] for i in whichExamples
                                     if i < len(tsFromDataset)]
                tsFromDataset = filter(
                    lambda ts: len(ts.labels) >= instancesPerTs, tsFromDataset)
                tsList += tsFromDataset
        else:
            tsList = ucr.labeledTsListFromDataset(datasetName, instancesPerTs=instancesPerTs)
            if hasWhichExamples:
                tsList = [tsList[i] for i in whichExamples]

    # other
    else:
        raise ValueError("Couldn't load unrecognized dataset {}".format(datasetName))

    enforceMinNum = minNumInstances > 0
    enforceMaxNum = maxNumInstances > 0
    if enforceMinNum and enforceMaxNum:
        if maxNumInstances < minNumInstances:
            raise ValueError("maximum number of instances {} < minimum number {}".format(
                maxNumInstances, minNumInstances))

    # at least/most this many instances of at least one label
    if enforceMinNum or enforceMaxNum:
        newTsList = []
        for ts in tsList:
            uniqs, counts = np.unique(ts.labels, return_counts=True)
            if not len(counts):  # no labels at all
                if enforceMinNum:
                    continue
                else:
                    newTsList.append(ts)
                    continue
            highestCount = np.max(counts)
            if enforceMinNum and highestCount < minNumInstances:
                continue
            if enforceMaxNum and highestCount > maxNumInstances:
                continue
            newTsList.append(ts)
        tsList = newTsList

    # return only this much initial data
    # XXX: this is super unsafe as far as labels
    if cropDataLength > 0:
        for ts in tsList:
            length = len(ts.data)
            if length <= cropDataLength:
                continue

            # if seed was set, pick a random subsequence of the right length
            # XXX user should really have to request this explicitly
            if seed is None:
                ts.data = ts.data[:cropDataLength]
            else:
                diff = length - cropDataLength
                startIdx = int(np.random.random() * diff)
                endIdx = startIdx + cropDataLength

                ts.data = ts.data[startIdx:endIdx]

    return tsList


# @_memory.cache
def loadDatasets(datasetNames, *args, **kwargs):
    datasetNames = synth.ensureIterable(datasetNames)
    allTs = []
    for name in datasetNames:
        allTs += loadDataset(name, **kwargs)

    return allTs


class DataLoader(BaseEstimator, TransformerMixin):

    def __init__(self, datasetName='shapes', whichExamples=None, seed=None,
                 instancesPerTs=10, minNumInstances=2, maxNumInstances=None,
                 cropDataLength=None):
        self.datasetName = datasetName  # note that this can be a collection
        self.whichExamples = whichExamples
        self.seed = seed
        self.instancesPerTs = instancesPerTs
        self.minNumInstances = minNumInstances
        self.maxNumInstances = maxNumInstances
        self.cropDataLength = cropDataLength

    def fit(self, X, y=None, **params):
        # have to load dataset here, not in init, to
        # work with BaseEstimator cloning
        self.tsList_ = loadDatasets(
            self.datasetName,
            seed=self.seed,
            whichExamples=self.whichExamples,
            instancesPerTs=self.instancesPerTs,
            minNumInstances=self.minNumInstances,
            maxNumInstances=self.maxNumInstances,
            cropDataLength=self.cropDataLength)

        return self

    def transform(self, X):
        return self.tsList_
