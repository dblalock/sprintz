#!/usr/bin/env/python

# NOTE: this file may only support unix systems since it's running
# shell commands

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import soundfile as sf # http://pysoundfile.readthedocs.org/en/0.7.0/
import librosa # https://bmcfee.github.io/librosa/index.html

from joblib import Memory
memory = Memory('./', verbose=2)

import paths as pth
from ..utils import sequence as seq

SAMPLE_RATE_HZ = 20 * 1000 # actual sampling rate of data

# ================================================================
# Public funcs
# ================================================================

def getDataFilePaths():
	return _replaceExtensions(_getOrigDataFilePaths(), 'aif')


def getAllRecordings():
	allPaths = getDataFilePaths()
	return map(lambda path: Recording(path), allPaths)


def getPureRecordings():
	# filter out recordings with more than one pattern
	recordings = getAllRecordings()
	return filter(lambda r: len(np.unique(r.digits)) == 1, recordings)


@memory.cache
def pureRecordingsForDigits(): # dict: digit -> [recording]
	pureRecordings = getPureRecordings()
	digit2recordings = seq.splitElementsBy(lambda r: r.digits[0], pureRecordings)
	return digit2recordings


@memory.cache
def getConcatenatedRecordingsForDigits(instancesPerTs=10, singleDigitsOnly=False,
	examplesPerDigit=None, estimateBoundaries=True, enemyInstancesPerTs=0):
	digit2recordings = pureRecordingsForDigits()

	hasExamplesPerDigit = (examplesPerDigit is not None) and len(examplesPerDigit) > 0

	# we allow at most one instance of each other class so there's only one
	# repeating "pattern"
	numDigitClasses = len(digit2recordings)
	if enemyInstancesPerTs > numDigitClasses - 1:
		print("getConcatenatedRecordingsForDigits(): WARNING: "
			"enemyInstancesPerTs {} > num digits - 1; will be truncated".format(
				enemyInstancesPerTs))
		enemyInstancesPerTs = numDigitClasses

	combinedRecordings = []
	for digit, recordings in digit2recordings.iteritems():
		np.random.shuffle(recordings) # in-place shuffle
		if singleDigitsOnly:
			recordings = filter(lambda r: len(r.digits) == 1, recordings)
		numRecordings = len(recordings)

		groupNum = -1 # -1 so first group will be 0, to match indices
		group = []
		numDigitsSoFar = 0
		for i, r in enumerate(recordings):
			# if we have as many groups as the number requested, we're done
			if hasExamplesPerDigit and (groupNum + 1) == examplesPerDigit:
				break

			numDigitsSoFar += len(r.digits)
			group.append(r)
			thisIsLastRecording = i == numRecordings - 1
			if numDigitsSoFar >= instancesPerTs or thisIsLastRecording:
				groupNum += 1

				if groupNum % 10 == 0:
					print("creating recording for digit, group = {}, {}".format(
						digit, groupNum))

				otherDigits = digit2recordings.keys()
				otherDigits.remove(digit)
				if enemyInstancesPerTs > 0:
					whichOtherDigits = np.random.choice(otherDigits, enemyInstancesPerTs)
					for dgt in whichOtherDigits:
						whichRecording = np.random.choice(digit2recordings[dgt])
						group.append(whichRecording)
					np.random.shuffle(group)

				# compute flattened list of all digits
				allDigits = [r.digits for r in group]
				allDigits = [item for sublist in allDigits for item in sublist]
				# totalCount = sum([len(digits) for digits in allDigits])
				# allDigits = np.array([digit] * totalCount)

				allDataMats = [r.data for r in group]
				combinedData = np.vstack(allDataMats)

				# compute starts and ends of each orig recording in
				# combined recording
				dataLengths = np.array([len(r.data) for r in group])
				cumLengths = np.cumsum(dataLengths)
				recordingStartIdxs = np.append(0, cumLengths[:-1])
				if estimateBoundaries:
					# compute (estimated) start and end indices of utterances
					# within each recording
					allUtteranceStartEnds = [r.utteranceStartEnd(whichRepresentation='raw')
						for r in group]
					allUtteranceStarts, allUtteranceEnds = zip(*allUtteranceStartEnds)

					startIdxs = recordingStartIdxs + np.array(allUtteranceStarts)
					endIdxs = recordingStartIdxs + np.array(allUtteranceEnds)
				else:
					startIdxs = recordingStartIdxs
					endIdxs = cumLengths - 1

				# repeat starts and ends multiple times when there are
				# multiple utterances in a single recording; we shift them
				# forward slightly so that it's visually clear there's more
				# than one when we plot the boundaries, and so that there's
				# an unambiguous ordering thereof for matching purposes
				starts = []
				ends = []
				if estimateBoundaries:
					for j, r in enumerate(group):
						numDigits = len(r.digits)
						startIdx, endIdx = startIdxs[j], endIdxs[j]
						if numDigits == 1:
							starts.append(startIdx)
							ends.append(endIdx)
						elif numDigits >= 2:
							# assume individual digit utterances uniformly
							# spaced and contiguous within the interval (tends
							# to be a pretty good approximation)
							intervalFracs = np.arange(numDigits + 1) / float(numDigits)
							gap = endIdx - startIdx
							breaks = (gap * intervalFracs).astype(np.intp)
							utteranceStartIdxs = startIdx + breaks[:-1]
							utteranceEndIdxs = startIdx + breaks[1:]
							starts += list(utteranceStartIdxs)
							ends += list(utteranceEndIdxs)
						else:
							raise IndexError("Recording"
								"{} contained no digits!".format(r.name))
				else:
					multipleUtterancesStep = 2
					for j, r in enumerate(group):
						numDigits = len(r.digits)
						endOffset = multipleUtterancesStep * (numDigits - 1)
						for k in range(numDigits):
							offset = multipleUtterancesStep * k
							starts.append(startIdxs[j] + offset)
							ends.append(endIdxs[j] - endOffset + offset)
				startIdxs = np.array(starts)
				endIdxs = np.array(ends)

				samplerate = SAMPLE_RATE_HZ
				speakerName = "Combined-{}".format(groupNum) # just an identifier
				combinedRecording = Recording(speaker=speakerName, digits=allDigits,
					data=combinedData, samplerate=samplerate,
					startIdxs=startIdxs, endIdxs=endIdxs)
				combinedRecordings.append(combinedRecording)

				group = [] # reset group
				numDigitsSoFar = 0

	return combinedRecordings


# ================================================================
# Private funcs
# ================================================================

def _getOrigDataFilePaths():
	# path: data/{adults,children}/{train,test}/{man,woman}/subject/file.wav
	return glob.glob(pth.TIDIGITS + '/*/t*/*/*/*.wav')


def _replaceExtensions(paths, ext):
	return map(lambda p: p[:-3] + ext, paths) # clearly not robust at all


def _createDataFiles():
	inPaths = _getOrigDataFilePaths()
	outPaths = getDataFilePaths()

	for inPath, outPath in zip(inPaths, outPaths):
		# use the sph2pipe script from NIST to convert audio from
		# sphere format to aif format
		# XXX: only tested on OS X
		cmd = "{} -f aif {} > {}".format(pth.SPH2PIPE_EXECUTABLE, inPath, outPath)
		os.system(cmd)


def _ensureDirExists(dir):
	if not os.path.exists(dir):
		os.makedirs(dir)


def _extractFilenameFromPath(path):
	return os.path.basename(path).split('.')[0]


def _parseSpeakerFromPath(path):
	pathComponents = path.split(os.sep)
	return pathComponents[-2] # paths end in subjectId/fileName.ext


def _parseDigitsFromPath(path):
	fileName = _extractFilenameFromPath(path)
	if fileName[-1] in 'ab': # strip a or b for "production" type
		fileName = fileName[:-1]
	chars = [c for c in fileName]
	return np.array(chars, dtype=np.object) # return a string array


def _plotSpectrogram(X, title='Spectrogram', **kwargs):
	librosa.display.specshow(X.T, x_axis='time', **kwargs)
	plt.colorbar()
	plt.title(title)
	plt.tight_layout()


def _utteranceStartEnd(mfccs, thresh=.1):
	"""
	Estimate the indices at which the utterance starts and ends based on where
	the power first and last exceeds `thresh` * the maximum power in the signal
	"""
	power = mfccs[:, 0] # XXX assumes 1st coeff is power
	minPower, maxPower = np.min(power), np.max(power)
	thresh = thresh * (maxPower - minPower) + minPower
	aboveThreshIdxs = np.where(power > thresh)[0]
	# print "_utteranceStartEnd: returning {}, {} for mfccs of length {}".format(
	# 	aboveThreshIdxs[0], aboveThreshIdxs[-1], len(mfccs))
	return aboveThreshIdxs[0], aboveThreshIdxs[-1]


def _mfccIdxsToRawDataIdxs(idxs, dataLen=-1):
	if dataLen < 0:
		dataLen = np.inf
	if not hasattr(idxs, '__len__'):
		return min(idxs * RAW_SAMPLES_PER_MFCC_SAMPLE, dataLen - 1)
	idxs = np.asanyarray(idxs)
	idxs = (idxs * RAW_SAMPLES_PER_MFCC_SAMPLE).astype(np.intp)
	return np.minimum(idxs, dataLen - 1)


def _rawDataIdxsToMfccIdxs(idxs):
	if not hasattr(idxs, '__len__'):
		return idxs / RAW_SAMPLES_PER_MFCC_SAMPLE
	idxs = np.asanyarray(idxs)
	return (idxs / RAW_SAMPLES_PER_MFCC_SAMPLE).astype(np.intp)


# ================================================================
# Types
# ================================================================

# RECORDING_WINDOW_LEN_SECS = .025 # 25ms
RECORDING_WINDOW_LEN_SECS = .01 # 10ms
# RECORDING_WINDOW_SPACING_SECS = .01 # 10ms
RECORDING_WINDOW_SPACING_SECS = .005 # 5ms
RECORDING_FFT_LEN = 512
RECORDING_NUM_MFCCS = 13

# conversion factor: samples/sec * secs/sample
RAW_SAMPLES_PER_MFCC_SAMPLE = SAMPLE_RATE_HZ * RECORDING_WINDOW_SPACING_SECS


class Recording(object):

	def __init__(self, path=None, speaker=None, digits=None, data=None,
		samplerate=-1, startIdxs=None, endIdxs=None):
		self.path = path
		self.speaker = _parseSpeakerFromPath(path) if speaker is None else speaker
		self.digits = _parseDigitsFromPath(path) if digits is None else digits
		if path is not None:
			self._digitsName = _extractFilenameFromPath(path)
		else:
			self._digitsName = ''.join(self.digits) # XXX excludes production (a vs b)
			# just first digit for combined data for a single digit
			if len(np.unique(self.digits)) == 1:
				digit = self._digitsName[0]
				self._digitsName = "{}x{}".format(digit, len(self.digits))
		self._digitsName = self._digitsName.replace('o', 'O')
		self._digitsName = self._digitsName.replace('z', 'Z')

		self._samplerate = samplerate
		self._data = data
		self._startIdxs = startIdxs
		self._endIdxs = endIdxs

		self._mfccs = None
		self._fbank = None
		self._logfbank = None

	def _loadData(self):
		if self._data is None:
			self._data, self._samplerate = sf.read(self.path)

	def _windowLength(self):
		self._loadData()
		return int(RECORDING_WINDOW_LEN_SECS * self._samplerate)

	def _windowSpacing(self):
		self._loadData()
		return (RECORDING_WINDOW_SPACING_SECS * self._samplerate)

	def startEndIdxs(self, whichRepresentation='mfccs'):
		self._loadData()
		if self._startIdxs is None or self._endIdxs is None:
			return None, None

		if whichRepresentation == 'raw':
			return self._startIdxs, self._endIdxs
		elif whichRepresentation == 'mfccs':
			starts = _rawDataIdxsToMfccIdxs(self._startIdxs)
			ends = _rawDataIdxsToMfccIdxs(self._endIdxs) - 1
			return starts, ends
		else:
			raise NotImplementedError("Start and end idxs only supported"
				"for raw data and mfccs")

	def utteranceStartEnd(self, whichRepresentation='raw'):
		start, end = _utteranceStartEnd(self.mfccs)
		if whichRepresentation == 'mfcc':
			return start, end
		elif whichRepresentation == 'raw':
			return _mfccIdxsToRawDataIdxs((start, end), len(self.data))
		else:
			raise NotImplementedError("Can't find utterance start and end for"
				"unsupported representation {}".format(whichRepresentation))

	@property
	def name(self):
		return self.speaker + '_' + self._digitsName

	@property
	def data(self):
		self._loadData()
		return self._data

	@property
	def mfccs(self, *args, **kwargs):
		self._loadData()
		if (self._mfccs is None) or (args is not None) or (kwargs is not None):
			# self._mfccs = mfcc(self._data, self._samplerate, *args, **kwargs)
			self._mfccs = librosa.feature.mfcc(self._data.flatten(),
				self._samplerate, n_mfcc=RECORDING_NUM_MFCCS,
				n_fft=RECORDING_FFT_LEN,
				# n_fft=self._windowLength(),
				hop_length=self._windowSpacing(),
				# win_length=self._windowLength()
			)
			self._mfccs = self._mfccs.T
		return self._mfccs

	@property
	def fbank(self, *args, **kwargs):
		self._loadData()
		if (self._fbank is None) or (args is not None) or (kwargs is not None):
			# nfft = self._windowLength()
			# print self._windowLength()
			self._fbank = librosa.core.stft(self._data.flatten(),
				# n_fft=nfft,
				n_fft=RECORDING_FFT_LEN,
				win_length=self._windowLength(),
				hop_length=self._windowSpacing()
			)
			self._fbank = np.absolute(self._fbank.T)
		return self._fbank

	@property
	def logfbank(self, *args, **kwargs):
		return np.log(self.fbank(*args, **kwargs))

	def _generateFileNameNoExt(self):
		return "{}_{}".format(self._digitsName, self.speaker)

	def _generatePlotTitle(self):
		return self._digitsName

	def plot(self, raw=True, mfccs=False, fbank=False, logfbank=False,
		saveDir=None, fileExt='pdf'):

		if saveDir and fileExt[0] != '.':
			fileExt = '.' + fileExt
		fName = self._generateFileNameNoExt() + fileExt

		title = self._generatePlotTitle()

		if raw:
			plt.figure()
			plt.plot(self.data())
			plt.xlim((0, len(self.data())))
			plt.title(title)
			if saveDir:
				savePath = os.path.join(saveDir, 'raw')
				_ensureDirExists(savePath)
				savePath = os.path.join(savePath, fName)
				plt.savefig(savePath)
			plt.close()
		if mfccs:
			plt.figure(figsize=(10, 6))
			plt.plot(self.mfccs)
			plt.xlim((0, len(self.mfccs)))
			plt.title(title + " MFCCs")
			if saveDir:
				savePath = os.path.join(saveDir, 'mfccs')
				_ensureDirExists(savePath)
				savePath = os.path.join(savePath, fName)
				plt.savefig(savePath)
			plt.close()
		if fbank:
			plt.figure()
			plotTitle = title + " filterBankCoeffs"
			_plotSpectrogram(self.fbank, title=plotTitle,
				sr=self._samplerate, hop_length=self._windowSpacing())
			if saveDir:
				savePath = os.path.join(saveDir, 'fbank')
				_ensureDirExists(savePath)
				savePath = os.path.join(savePath, fName)
				plt.savefig(savePath)
			plt.close()
		if logfbank:
			plt.figure()
			plotTitle = title + " log filterBankCoeffs"
			_plotSpectrogram(self.logfbank, title=plotTitle,
				sr=self._samplerate, hop_length=self._windowSpacing())
			if saveDir:
				savePath = os.path.join(saveDir, 'logfbank')
				_ensureDirExists(savePath)
				savePath = os.path.join(savePath, fName)
				plt.savefig(savePath)
			plt.close()


# ================================================================
# Main
# ================================================================

def main():
	# _createDataFiles() # only need to do this once
	# return

	np.random.seed(12345)
	import datasets
	# saveDir = 'figs/tidigits/concat3/'
	saveDir = 'figs/tidigits/concat4/'
	tsList = datasets.loadDataset('tidigits_grouped_mfcc', whichExamples=range(10))
	print '------------------------'
	for ts in tsList:
		ts.plot(saveDir)
		print ts.name, ts.labels
	return

	# saveDir = 'figs/tidigits/'
	# saveDir = 'figs/tidigits/concat/'
	saveDir = 'figs/tidigits/concat2/'
	# for r in getAllRecordings()[:3]:
	# for r in getAllRecordings():
	# for r in getConcatenatedRecordingsForDigits():
	for r in getConcatenatedRecordingsForDigits(examplesPerDigit=1):
		print("plotting recording: {}".format(r.name))

		data = r.data
		print len(data)
		print len(r.mfccs)

		r._data = np.vstack((data, data))
		print len(r._data)
		print len(r.mfccs(dummyKwarg=True)) # refresh mfccs

		# print "digits: ", r.digits[:10]
		# r.plot(True, True, True, True, saveDir=saveDir)
		# r.plot(True, True, saveDir=saveDir)
		# r.plot(raw=False, mfccs=True, fbank=False, logfbank=True, saveDir=saveDir)
		# r.plot(raw=False, mfccs=True, saveDir=saveDir)
		# r.plot(saveDir=saveDir)

	return

	paths = getDataFilePaths()
	paths = paths[:3]
	for path in paths:
		data, samplerate = sf.read(path)
		print data.shape, samplerate

		# plt.plot(data)

		# mfccs = mfcc(data, samplerate, winstep=.005, ceplifter=0)
		# mfccs = mfcc(data, samplerate)
		mfccs2 = librosa.feature.mfcc(data.flatten(), samplerate,
			n_mfcc=13, n_fft=512, hop_length=(.01*samplerate))
		# print mfccs.shape, mfccs2.shape

		# add on first and second discrete derivatives
		firstDiffs = np.diff(mfccs2, axis=1)
		secondDiffs = np.diff(mfccs2, axis=1, n=2)
		nCoeffs = mfccs2.shape[0]
		nSamples = mfccs2.shape[1]
		allSignals = np.zeros((nCoeffs * 3, nSamples))
		allSignals[:nCoeffs, 1:] = firstDiffs
		allSignals[nCoeffs:2*nCoeffs, 2:] = secondDiffs

		# allSignals = np.vstack((mfccs2, firstDiffs, secondDiffs))

		# plt.figure()
		# plt.plot(mfccs)

		# plt.figure()
		# plt.plot(mfccs2.T)

		stft = librosa.core.stft(data.flatten(), n_fft=512,
			win_length=(.01*samplerate))
		print stft.shape

		plt.figure()
		librosa.display.specshow(stft, x_axis='time')
		plt.colorbar()
		plt.title('STFT')
		plt.tight_layout()

		# plt.figure()
		# plt.plot(allSignals.T)

		# okay, so I have no idea what python_speech_features is
		# doing, but its output looks like garbage and librosa's
		# looks nice

		# plt.figure()
		# librosa.display.specshow(mfccs.T, x_axis='time')
		# plt.colorbar()
		# plt.title('MFCC')
		# plt.tight_layout()

		# plt.figure()
		# librosa.display.specshow(mfccs2, x_axis='time')
		# plt.colorbar()
		# plt.title('MFCC')
		# plt.tight_layout()

	plt.show()


if __name__ == '__main__':
	main()
