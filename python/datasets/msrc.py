#!/bin/python

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import paths
from ..utils.files import ensureDirExists

NUM_RECORDINGS = 594
DATA_DIR = paths.MSRC_12
# OUTPUT_DATA_DIR = './data'

join = os.path.join

FIG_SAVE_DIR = join('figs','msrc')
SAVE_DIR_LINE_GRAPH = join(FIG_SAVE_DIR, 'line')
SAVE_DIR_IMG = join(FIG_SAVE_DIR, 'img')

# Notes:
# 	-Tagstream times are sort of in the middle of the gesture, but
# 	not really; they're sometimes very close to the beginning or
# 	the end; basically, they don't seem to be super-consistent,
# 	objective ground truth
# 	-Many of these recordings include a bunch of extra material
# 	(in some cases even 10 instances of two totally different
# 	things at different times...)
#	-The number of gestures in a given recording varies; typically
#	10, but plotting recordings reveals numbers from 8-13
# 	-It's not clear what every 4th column of the data matrix is;
#	columns 1-3 + 4k, k = {0..19} are x,y,z position, but I can't
# 	find what the remaining fourth of the columns are...

CLASS_IDS_2_NAMES = {
	1: "Start system",
	2: "Duck",
	3: "Push right",
	4: "Goggles",
	5: "Wind it up",
	6: "Shoot",
	7: "Bow",
	8: "Throw",
	9: "Had enough",
	10: "Change weapon",
	11: "Beat both arms",
	12: "Kick"
}


def getFileNames():
	# get paths of all data files
	dataFiles = glob.glob('%s/*.csv' % DATA_DIR)
	tagstreamFiles = glob.glob('%s/*.tagstream' % DATA_DIR)

	# there should be 594 of each
	assert(len(dataFiles) == NUM_RECORDINGS), \
		"missing data files: should be %d in %s" \
		% (NUM_RECORDINGS, DATA_DIR)
	assert(len(tagstreamFiles) == NUM_RECORDINGS), \
		"missing tagstream files: should be %d in %s" \
		% (NUM_RECORDINGS, DATA_DIR)

	return (dataFiles, tagstreamFiles)


def parseTime(time):
	# I have no idea what their time stamps mean, but this magic
	# line from their matlab code converts things to...microseconds?
	return (int(time)*1000 + 49875/2)/49875


def parseFileName(fName):
	# file names are of form P#_#_#[A]_P#.csv
	# print("reading file: %s" % fName)
	fName = fName.split("/")[-1]
	# print("file name: %s" % fName)
	noSuffix = fName.split(".")[0]
	fields = noSuffix.split("_")
	instruction = ''.join(fields[0:2])  # form is P%d_%d - no clear mapping to
										# actual instruction modalities...
	gestureId = fields[2]				# form is %d
	subjId = int(fields[3][1:])			# form is p%d
	if (gestureId[-1] == 'A'):
		twoModalities = True
		gestureId = gestureId[:-1]
	else:
		twoModalities = False
	gestureId = int(gestureId)

	return (instruction, gestureId, subjId, twoModalities)


def readAnswerTimes(tagFile):
	times = []
	with open(tagFile, 'r') as f:
		f.readline()  	# first line is garbage, not data
		for line in f:
			time, gesture = line.split(';')
			times.append(parseTime(time))
	return times


def readDataFile(dataFile):
	contents = np.genfromtxt(dataFile, delimiter=' ')
	timeStamps = contents[:,0]
	data = contents[:,1:]
	rowSums = np.sum(data,1)
	nonZeroRowIdxs = np.where(rowSums != 0)
	data = data[nonZeroRowIdxs]
	timeStamps = timeStamps[nonZeroRowIdxs]	 # no need to convert these
	# print(data.shape)
	# print(timeStamps.shape)
	assert data.shape[0] == timeStamps.shape[0]
	return data, timeStamps


def getRecordings(idxs=None):
	dataFiles, tagFiles = getFileNames()
	if idxs is None:
		idxs = range(len(dataFiles))
	for i in idxs:
		try:
			r = Recording(dataFiles[i], tagFiles[i], recID=i)
			yield r
		except IndexError: # empty or all 0s file -> IndexError
			print("skipping broken recording #{}".format(i))
			continue


def _compute_label_idxs(labelTimes, sampleTimes):
	labelIdxs = np.empty(len(labelTimes), dtype=np.int)
	for i, time in enumerate(labelTimes):
		labelIdxs[i] = np.where(sampleTimes >= time)[0][0] # extra [0] to unpack where()
	return labelIdxs


# def _uniformlyResample1D(x, y):
# 	interpFunc = interp.interp1d(x, y)
# 	uniformX = np.linspace(np.min(x), np.max(x), len(x))
# 	return interpFunc(uniformX)

# def _uniformlyResample(x, Y):
# 	Ynew = np.empty(Y.T.shape)
# 	for i, col in enumerate(Y.T):
# 		Ynew[i] = _uniformlyResample1D(x, col)
# 	return Ynew.T


class Recording:
	def __init__(self, dataFile, tagFile, recID=-1):
		print("creating recording #{}".format(recID))
		self.id = recID
		self.fileName = dataFile.split('.')[0]
		self.instruction, self.gestureId, self.subjId, self.twoModalities = \
			parseFileName(dataFile)
		self.gestureLabel = CLASS_IDS_2_NAMES[self.gestureId]
		self.gestureTimes = readAnswerTimes(tagFile)

		self.data, self.sampleTimes = readDataFile(dataFile)
		# self.rawData, self.sampleTimes = readDataFile(dataFile)
		# self.data = _uniformlyResample(self.sampleTimes, self.rawData)

		self.gestureIdxs = _compute_label_idxs(self.gestureTimes, self.sampleTimes)

	def __str__(self):
		s1 = "instruction: %s\ngestureId: %s\nsubjId: %s\n" \
			% (self.instruction, self.gestureId, self.subjId)
		s2 = "data sz: %d x %d" % self.data.shape
		s3 = str(self.gestureTimes) + str(self.data)
		return s1 + s2 + s3

	def plot(self, saveDir=None):
		# for i in range(3):
			# plt.plot(self.sampleTimes, self.data[:,i])
		plt.figure()
		plt.autoscale(tight=True)
		plt.plot(self.sampleTimes, self.data)
		minVal = np.min(self.data) - .2
		maxVal = np.max(self.data) + .2
		for time in self.gestureTimes:
			plt.plot([time, time], [minVal, maxVal],
				color='k', linestyle='--', linewidth=1)
		plt.title(self.gestureLabel)
		if saveDir:
			ensureDirExists(saveDir)
			fileName = "%s_%d" % (self.gestureLabel, self.id)
			fileName = join(saveDir, fileName)
			plt.savefig(fileName)
		# else:
		# 	plt.show() # TODO uncomment


def main():
	dataFiles, tagFiles = getFileNames()
	# for i in range(len(dataFiles)):
	for i in range(0,550,3):  # just an arbitrary, evenly-spaced, subset
	# for i in range(11,550,40):  # just an arbitrary, evenly-spaced, subset
	# for i in range(11,550,500):  # just an arbitrary, evenly-spaced, subset
	# for i in range(5, 15, 2):
		r = Recording(dataFiles[i], tagFiles[i], i)
		# print(r)
		r.plot(saveDir=SAVE_DIR_LINE_GRAPH)
		# r.plot()
	# plt.show()

	# TODO save imgs
	# TODO move imshow() from pamap stuff to shared file and img this data

	print("Done")

if __name__ == "__main__":
	main()
