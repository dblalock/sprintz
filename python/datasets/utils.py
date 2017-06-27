#!/usr/bin/env/python

import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle

DEFAULT_LABEL = 0

from synthetic import concatWithPadding, ensure2D
from ..utils.sequence import splitElementsBy, splitIdxsBy

# ================================================================ Plotting

def plotVertLine(x, ymin=None, ymax=None, ax=None, **kwargs):
	if ax and (not ymin or not ymax):
		ymin, ymax = ax.get_ylim()
	if not ax:
		ax = plt

	kwargs['color'] = kwargs.get('color') or 'k'
	kwargs['linestyle'] = kwargs.get('linestyle') or '--'
	kwargs['linewidth'] = kwargs.get('linewidth') or 2

	ax.plot([x, x], [ymin, ymax], **kwargs)


def plotRect(ax, xmin, xmax, ymin=None, ymax=None, alpha=.2,
	showBoundaries=True, color='grey', fill=True, **kwargs):
	if ax and (ymin is None or ymax is None):
		ymin, ymax = ax.get_ylim()
	if fill:
		ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
			facecolor=color, alpha=alpha))
	if showBoundaries:
		plotVertLine(xmin, ymin, ymax, ax=ax, color=color, **kwargs)
		plotVertLine(xmax, ymin, ymax, ax=ax, color=color, **kwargs)


# ================================================================ Animation

def animateSubseqs(X, windowLen, step=None, figsize=None, dataName=None,
	ylimits=None, idxOffsetTitle=0, idxOffsetXLabel=0,
	rangeStartIdxs=None, rangeEndIdxs=None,
	rangeLabels=None):
	"""plots the data in each sliding window position, and creates an
	animation from the sequence of plots. If rangeStartIdxs and rangeEndIdxs
	are specified, it will mark the associated regions in the plots, also
	writing the content of the corresponding entry of rangeLabels in each

	Parameters
	----------

	X: 2d array
		the data to plot; each row is a sample
	windowLen: int
		length of the sliding window
	step: int
		the increment by which to slide the window
	figsize: (x, y) tuple
		the size of the figure in inches
	dataName: string
		displayed in the title
	ylimits: (int, int)
		ylimits for each plots
	idxOffsetTitle: int
		value to add to the indices within each; if this is, say, the data
		starting at index 1000 from a longer dataset, this param can be used
		to add 1000 to what's displayed in the title
	idxOffsetXLabel: int
		like idxOffsetTitle, but adjusts the values displayed along the x axis;
		more useful since x values above 100,000 will be displayed in
		scientific notation
	rangeStartIdxs: vector of int
		the start indices of any labeled ranges in the data; will be marked with
		vertical lines
	rangeEndIdxs: vector of int
		the (non-inclusive) end indices of any labeled ranges in the data;
		will be marked with vertical lines; must be the same length as
		rangeStartIdxs if either of them are provided
	rangeLabels: vector of string
		the labels for each of the ranges specified by rangeStartIdxs and
		rangeEndIdxs; must be the same length as these if any of them are
		provided

	Returns
	-------
	A matplotlib Animation object
	"""

	X = np.asarray(X)
	if step < 1:
		step = windowLen / 4
	if ylimits is None:
		ylimits = [np.min(X), np.max(X)]
	yMin, yMax = ylimits[0], ylimits[1]

	dataName = dataName + ", " if dataName else ""

	# determine start locations for the window of data shown in each frame
	lastPossibleStartIdx = len(X) - windowLen
	windowStartIdxs = np.arange(0, lastPossibleStartIdx + 1, step)
	if windowStartIdxs[-1] != lastPossibleStartIdx: # ensure we don't miss a section
		windowStartIdxs = np.r_[windowStartIdxs, lastPossibleStartIdx]

	fig = plt.figure(figsize=figsize)
	ax = plt.gca()

	def animateFunc(frameNum):
		ax.cla()

		# ------------------------ plot data

		# show different indices for title and x axis so that x axis fits and
		# doesn't get shortened into illegible scientific notation
		startIdx = windowStartIdxs[frameNum]
		endIdx = startIdx + windowLen
		startIdxTitle = startIdx + idxOffsetTitle
		endIdxTitle = endIdx + idxOffsetTitle
		startIdxXlabel = startIdx + idxOffsetXLabel
		endIdxXlabel = endIdx + idxOffsetXLabel

		ax.set_title("{0}{1}-{2}".format(dataName, startIdxTitle, endIdxTitle))
		ax.plot(np.arange(startIdxXlabel, endIdxXlabel), X[startIdx:endIdx])
		ax.set_ylim(yMin, yMax)
		ax.set_xlim(startIdxXlabel, endIdxXlabel)

		# ------------------------ plot labeled ranges

		# if no labeled ranges provided, we're done
		if (rangeStartIdxs is None) or (len(rangeStartIdxs) < 1):
			return

		windowRangeStartIdx, windowRangeEndIdx = whereStartEndPairsInRange(
			rangeStartIdxs, rangeEndIdxs, startIdx, endIdx)

		# if no labeled ranges in this window, we're done
		if windowRangeEndIdx <= windowRangeStartIdx:
			return

		# print("--- window #{}: {}-{}".format(frameNum, startIdxTitle, endIdxTitle))

		for i in range(windowRangeStartIdx, windowRangeEndIdx):
			label = str(rangeLabels[i])
			ts, te = rangeStartIdxs[i], rangeEndIdxs[i]
			ts, te = ts + idxOffsetXLabel, te + idxOffsetXLabel

			if ts < 0 or te > endIdxXlabel: # can happen since only end idxs monotonic
				continue

			plotRect(ax, ts, te, fill=False) # no fill because it doesn't clear

			# position label near start, but shifted based on width to
			# differentiate ranges with the same start; also, stagger heights
			# so that labels don't end up on top of one another
			x = ts + (te - ts) / 10
			yFrac = .67
			yFrac += .04 * (i // 1 % 2)
			yFrac += .08 * (i // 2 % 2)
			yFrac += .16 * (i // 4 % 2)
			y = yFrac * (yMax - yMin) + yMin
			ax.annotate(label, xy=(x, y), xycoords='data')

			# print("{}: {}-{}\t({}-{})\t\tx={:d}\tlbl={}".format(label, ts, te,
			# 	rangeStartIdxs[i], rangeEndIdxs[i], x, label))

	plt.close()

	return animation.FuncAnimation(fig, animateFunc, frames=len(windowStartIdxs), blit=False)


def saveAnimation(anim, path, fps=25):
	anim.save(path, writer='ffmpeg', fps=fps) # Note: ensure that ffmpeg is installed


def generateVideos(ar, dataName="Data", step=5000, windowLen=300,
	epochSz=100000, saveInDir=None, rangeStartIdxs=None, rangeEndIdxs=None,
	rangeLabels=None, ylimits=None, createSubdir=True):
	"""Given a long time series ar, plots subsequences of length windowLen,
	in videos spanning step samples. epochSz is used to sort the videos into
	subdirectories. The range{Start,End}Idxs and rangeLabels are used to
	plot an arbitrary set of labeled ranges in the data."""

	# sanity check labeled ranges
	hasRanges = rangeStartIdxs is not None or rangeEndIdxs is not None or rangeLabels is not None
	hasRanges = hasRanges and len(rangeStartIdxs) > 0
	if hasRanges:
		assert(len(rangeStartIdxs) == len(rangeEndIdxs))
		if rangeLabels is not None:
			assert(len(rangeStartIdxs) == len(rangeLabels))

	# determine window length
	if windowLen is None or windowLen < 0:
		windowLen = int(len(ar) / len(rangeStartIdxs))
	elif windowLen <= 1.:
		windowLen = int(windowLen * len(ar))
	# determine video length
	if step is None or step < 0:
		step = len(ar) + 1 # put everything in one video
	elif step <= 1.:
		step = int(step * len(ar))

	startIdxsInVid = None
	endIdxsInVid = None
	labelsInVid = None

	lastRangeIdx = 0
	for epochNum, epochStartIdx in enumerate(range(0, len(ar), epochSz)):
		epochEndIdx = epochStartIdx + epochSz
		epochData = ar[epochStartIdx:epochEndIdx]

		if saveInDir:
			subdirName = dataName
			if not createSubdir:
				subdirName = ""
			if len(ar) > epochSz:
				subdir = "{0}k-{1}k".format(epochStartIdx / 1000, epochEndIdx / 1000)
				saveDir = os.path.join(saveInDir, subdirName, subdir)
			else:
				saveDir = os.path.join(saveInDir, subdirName)
			if not os.path.exists(saveDir):
				os.makedirs(saveDir)

		n = len(epochData)
		for startIdx in range(0, n, step): # for each video segment
			endIdx = min(startIdx + step, n)
			absoluteStartIdx = epochStartIdx + startIdx
			absoluteEndIdx = epochStartIdx + endIdx
			data = epochData[startIdx:endIdx]

			print "generating vid for section {0}-{1}".format(absoluteStartIdx, absoluteEndIdx)

			# figure out what labeled ranges are present in this video
			if hasRanges:
				rangeStartIdxs = rangeStartIdxs[lastRangeIdx:]
				rangeEndIdxs = rangeEndIdxs[lastRangeIdx:]
				rangeLabels = rangeLabels[lastRangeIdx:]
				first, last = whereStartEndPairsInRange(
					rangeStartIdxs, rangeEndIdxs, absoluteStartIdx, absoluteEndIdx)

				if (first >= 0) and (last > first):
					firstRangeIdx, lastRangeIdx = first, last
					startIdxsInVid = rangeStartIdxs[firstRangeIdx:lastRangeIdx] - absoluteStartIdx
					endIdxsInVid = rangeEndIdxs[firstRangeIdx:lastRangeIdx] - absoluteStartIdx
					labelsInVid = rangeLabels[firstRangeIdx:lastRangeIdx]
				else:
					lastRangeIdx = 0 # didn't move forward in range list
					startIdxsInVid = None
					endIdxsInVid = None
					labelsInVid = None

			print("labeled ranges in vid:")
			print np.c_[startIdxsInVid, endIdxsInVid, labelsInVid]

			figName = dataName + "_{0}-{1}".format(absoluteStartIdx, absoluteEndIdx-1)
			figPath = os.path.join(saveDir, figName + '.mp4')

			anim = animateSubseqs(data, windowLen, figsize=(8,6),
				dataName=dataName.title(), ylimits=ylimits,
				idxOffsetTitle=absoluteStartIdx, idxOffsetXLabel=startIdx,
				rangeStartIdxs=startIdxsInVid, rangeEndIdxs=endIdxsInVid,
				rangeLabels=labelsInVid)
			if saveInDir:
				saveAnimation(anim, figPath, fps=25)


# ================================================================ Annotations

def whereStartEndPairsInRange(startIdxs, endIdxs, minStartIdx, maxEndIdx):
	"""Given an ordered collection of start and end pairs, a minimum start
	index, and a maximum end index, return the first and last index into the
	start and end pairs collection such that the pair at that index is included
	in the range (the returned last index is not inclusive, as is typical in
	python).

	Note that this assumes that both the startIdxs and endIdxs are already
	sorted in ascending order. This is necessary for the returned indices
	to be meaningful when used to index the arrays passed in.

	Returns (-1, -1) if no (start, end) pairs fall in the range

	>>> starts = [0, 5]
	>>> ends = [1, 10]
	>>> minStart, maxEnd = 0, 20
	>>> whereStartEndPairsInRange(starts, ends, minStart, maxEnd)
	(0, 2)
	>>> minStart = 1
	>>> whereStartEndPairsInRange(starts, ends, minStart, maxEnd)
	(1, 2)
	>>> minStart, maxEnd = 0, 8
	>>> whereStartEndPairsInRange(starts, ends, minStart, maxEnd)
	(0, 1)
	>>> minStart, maxEnd = 1, 8
	>>> whereStartEndPairsInRange(starts, ends, minStart, maxEnd)
	(-1, -1)
	"""
	# fail fast
	assert(len(startIdxs) == len(endIdxs))
	empty = (-1, -1)
	if not len(startIdxs):
		return empty

	# find first startIdx >= minStartIdx
	tsIdx = -1
	ts = -1
	while ts < minStartIdx:
		try:
			tsIdx += 1
			ts = startIdxs[tsIdx]
		except IndexError: # reached end
			break

	# find last endIdx < maxEndIdx
	teIdx = tsIdx - 1
	te = -1
	while te < maxEndIdx:
		try:
			teIdx += 1
			te = endIdxs[teIdx]
		except IndexError:
			break

	if tsIdx == teIdx: # empty set
		return empty

	return tsIdx, teIdx


def unionOfRanges(rangeStartIdxs, rangeEndIdxs, n, padLen=0):
	"""
	>>> starts = [0]
	>>> ends = [10]
	>>> n = 10
	>>> padLen = 0
	>>> unionOfRanges(starts, ends, n, padLen)
	array([[ 0, 10]])
	>>>
	>>> starts = [2]
	>>> ends = [5]
	>>> unionOfRanges(starts, ends, n, padLen)
	array([[2, 5]])
	>>>
	>>> starts = [1, 4]
	>>> ends = [3, 7]
	>>> unionOfRanges(starts, ends, n, padLen) # doctest: +NORMALIZE_WHITESPACE
	array([[1, 3],
		[4, 7]])
	>>>
	>>> starts = [1, 4]
	>>> ends = [5, 6]
	>>> unionOfRanges(starts, ends, n, padLen) # doctest: +NORMALIZE_WHITESPACE
	array([[1, 6]])
	>>>
	>>> starts = [1, 3]
	>>> ends = [4, 7]
	>>> padLen = 1
	>>> unionOfRanges(starts, ends, n, padLen) # doctest: +NORMALIZE_WHITESPACE
	array([[0, 8]])
	>>>
	>>> n = 7
	>>> unionOfRanges(starts, ends, n, padLen) # doctest: +NORMALIZE_WHITESPACE
	array([[0, 7]])
	>>>
	>>> starts = [0, 6, 10, 20]
	>>> ends = [4, 9, 15, 30]
	>>> n = 100
	>>> unionOfRanges(starts, ends, n, padLen) # doctest: +NORMALIZE_WHITESPACE
	array([[ 0, 5],
		[ 5, 16],
		[19, 31]])
	>>>
	>>> starts = starts[::-1]
	>>> ends = ends[::-1]
	>>> unionOfRanges(starts, ends, n, padLen) # doctest: +NORMALIZE_WHITESPACE
	array([[ 0, 5],
		[ 5, 16],
		[19, 31]])
	>>>
	>>> padLen = 5
	>>> starts = [5, 10, 10]
	>>> ends = [20, 20, 30]
	>>> unionOfRanges(starts, ends, n, padLen) # doctest: +NORMALIZE_WHITESPACE
	array([[ 0, 35]])
	"""
	# sanity check args (just via asserts due to tactical laziness)
	assert(len(rangeStartIdxs) > 0)
	assert(len(rangeStartIdxs) == len(rangeEndIdxs))
	assert(np.all(rangeStartIdxs <= rangeEndIdxs))
	assert(np.min(rangeStartIdxs) >= 0)
	assert(np.max(rangeEndIdxs) <= n)
	assert(np.min(rangeEndIdxs) >= padLen)

	rangeStartIdxs = np.asarray(rangeStartIdxs)
	rangeEndIdxs = np.asarray(rangeEndIdxs)

	# sort by endIdx
	if len(rangeEndIdxs) > 1:
		sortIdxs = np.argsort(rangeEndIdxs)
		rangeStartIdxs = rangeStartIdxs[sortIdxs]
		rangeEndIdxs = rangeEndIdxs[sortIdxs]

	# compute earliest start time remaining (including current start time); eg,
	# 	[1,1,3,3,2,4,5,4,6] -> [1 1 2 2 2 4 4 4 6]
	# 	[1,1,3,3,0,4,5,4,6] -> [0 0 0 0 0 4 4 4 6]
	earliestFutureStarts = np.minimum.accumulate(rangeStartIdxs[::-1])[::-1]
	earliestFutureStarts -= padLen
	earliestFutureStarts = np.maximum(0, earliestFutureStarts)

	sectionStartIdxs = []
	sectionEndIdxs = []

	numInstances = len(rangeStartIdxs)
	currentSectionStart = earliestFutureStarts[0] # first start time anywhere
	for i in range(numInstances-1):
		te = rangeEndIdxs[i] + padLen
		earliestFutureStart = earliestFutureStarts[i+1]
		if currentSectionStart < te <= earliestFutureStart:
			sectionStartIdxs.append(currentSectionStart)
			sectionEndIdxs.append(te)
			currentSectionStart = earliestFutureStart

	finalEndIdx = min(rangeEndIdxs[-1] + padLen, n)
	sectionStartIdxs.append(currentSectionStart)
	sectionEndIdxs.append(finalEndIdx)

	return np.c_[sectionStartIdxs, sectionEndIdxs].astype(np.int)


def adjustedAnnotationIdxs(rangeStartIdxs, rangeEndIdxs, n, padLen=0):
	"""given that we're extracting the data around the annotations,
	adjust the start and end indices so that they're correct for the
	extracted data

	>>> starts = [0]
	>>> ends = [10]
	>>> n = 10
	>>> padLen = 0
	>>> adjustedAnnotationIdxs(starts, ends, n, padLen) # doctest: +NORMALIZE_WHITESPACE
	array([[ 0, 10]])
	>>>
	>>> starts = [5]
	>>> ends = [7]
	>>> adjustedAnnotationIdxs(starts, ends, n, padLen) # doctest: +NORMALIZE_WHITESPACE
	array([[0, 2]])
	>>>
	>>> starts = [5, 20]
	>>> ends = [7, 25]
	>>> n = 100
	>>> adjustedAnnotationIdxs(starts, ends, n, padLen) # doctest: +NORMALIZE_WHITESPACE
	array([[0, 2],
		[2, 7]])
	>>>
	>>> starts = [5, 10, 15]
	>>> ends = [7, 12, 18]
	>>> adjustedAnnotationIdxs(starts, ends, n, padLen) # doctest: +NORMALIZE_WHITESPACE
	array([[0, 2],
		[2, 4],
		[4, 7]])
	>>>
	>>> starts = [5, 6, 15]
	>>> ends = [7, 10, 18]
	>>> adjustedAnnotationIdxs(starts, ends, n, padLen) # doctest: +NORMALIZE_WHITESPACE
	array([[0, 2],
		[1, 5],
		[5, 8]])
	>>>
	>>> padLen = 1
	>>> adjustedAnnotationIdxs(starts, ends, n, padLen) # doctest: +NORMALIZE_WHITESPACE
	array([[ 1, 3],
		[ 2, 6],
		[ 8, 11]])
	"""

	if padLen is not None:
		assert(padLen >= 0) # could make sense, but unsupported in this func

	rangeStartIdxs = np.asarray(rangeStartIdxs)
	rangeEndIdxs = np.asarray(rangeEndIdxs)

	# if len(rangeStartIdxs) > 1:
	# 	sortIdxs = np.argsort(rangeEndIdxs)
	# 	rangeStartIdxs = rangeStartIdxs[sortIdxs]
	# 	rangeEndIdxs = rangeEndIdxs[sortIdxs]

	ranges = unionOfRanges(rangeStartIdxs, rangeEndIdxs, n, padLen)
	rangeStarts, rangeEnds = ranges[:, 0], ranges[:, 1]

	# compute number of points skipped; ie, skippedLengths[i] is the length
	# of all data before the ith range that won't be extracted
	rangeLengths = rangeEnds - rangeStarts
	skippedLengths = rangeEnds - np.cumsum(rangeLengths)

	newStartIdxs = []
	newEndIdxs = []
	numInstances = len(rangeStartIdxs)
	inWhichRange = 0
	for i in range(numInstances):
		ts, te = rangeStartIdxs[i], rangeEndIdxs[i]

		# find which combined range this original (start, end) pair lies in
		while rangeEnds[inWhichRange] < te:
			inWhichRange += 1
		skippedLen = skippedLengths[inWhichRange]

		newStartIdxs.append(ts - skippedLen)
		newEndIdxs.append(te - skippedLen)

	return np.c_[newStartIdxs, newEndIdxs].astype(np.int)


def dataNearAnnotations(X, rangeStartIdxs, rangeEndIdxs, padLen=0):
	"""extract (and concatenate) all the sections of X within padLen of any of
	the ranges defined by rangeStartIdxs and rangeEndIdxs"""
	combinedRanges = unionOfRanges(rangeStartIdxs, rangeEndIdxs, len(X), padLen)
	newIdxs = adjustedAnnotationIdxs(rangeStartIdxs, rangeEndIdxs, len(X), padLen)
	newStartIdxs, newEndIdxs = newIdxs[:, 0], newIdxs[:, 1]

	# print "dataNearAnnotations(): oldIdxs -> newIdxs"
	# for i in range(len(newIdxs)):
	# 	print np.r_[rangeStartIdxs[i], rangeEndIdxs[i]], "->", newIdxs[i]

	keepIdxs = []
	for i in range(len(combinedRanges)):
		dataStart = combinedRanges[i, 0] # has padding built in
		dataEnd = combinedRanges[i, 1]
		idxsInRange = range(dataStart, dataEnd)
		keepIdxs += list(idxsInRange)
		# print("{}-{}\t".format(dataStart, dataEnd))

	return X[np.array(keepIdxs)], newStartIdxs, newEndIdxs


def groupsOfAnnotationIdxsForLabels(labels, groupSize=10, shuffle=False):
	"""[label] -> (dict: label -> [idxs])
	Ie, return a list of lists of indices where a particular labels happens,
	with the inner lists of size groupSize. The point is to pull out groups
	of, say, 10 indices, where a given label occurs so we can subsequently
	pull out these sections of data.
	"""

	groupedLabels = splitIdxsBy(lambda lbl: lbl, labels)

	lbl2idxGroups = {}
	for lbl in groupedLabels:
		idxs = groupedLabels[lbl]
		idxGroups = formGroupsOfSize(idxs, groupSize=groupSize, shuffle=shuffle)
		lbl2idxGroups[lbl] = idxGroups

	return lbl2idxGroups


def sectionsOfDataNearAnnotations(X, startIdxs, endIdxs, labels,
	instancesPerTs=10, shuffle=False, padLen=0, keepLabels=None,
	datasetName="Dataset"):

	lbl2idxGroups = groupsOfAnnotationIdxsForLabels(labels,
		groupSize=instancesPerTs, shuffle=shuffle)

	tsList = []
	for lbl in lbl2idxGroups:
		if keepLabels and not (lbl in keepLabels): # only keep certain labels
			continue
		idxGroups = lbl2idxGroups[lbl]
		for groupNum, groupIdxs in enumerate(idxGroups):
			groupStartIdxs = startIdxs[groupIdxs]
			groupEndIdxs = endIdxs[groupIdxs]
			groupLabels = labels[groupIdxs]

			data, newStartIdxs, newEndIdxs = dataNearAnnotations(X,
				groupStartIdxs, groupEndIdxs, padLen=padLen)

			name = "{}-class{}-group{}".format(datasetName, lbl, groupNum)
			uniqId = hash(name)

			ts = LabeledTimeSeries(data, startIdxs=newStartIdxs,
				endIdxs=newEndIdxs, labels=groupLabels, name=name, id=uniqId)
			tsList.append(ts)

	return tsList


# like above, but allows multiple labels in a given section
def sectionsOfDataNearAnnotationsImpure(X, startIdxs, endIdxs, labels,
	instancesPerTs=10, shuffle=False, padLen=0, maxPadJitter=0,
	keepLabels=None, datasetName="Dataset"):

	assert(len(startIdxs) == len(endIdxs))
	assert(len(startIdxs) == len(labels))

	startIdxs = np.asarray(startIdxs)
	endIdxs = np.asarray(endIdxs)

	# filter out labels we don't care about
	if keepLabels:
		allIdxs = np.arange(len(labels))
		keepIdxs = [i for i in allIdxs if labels[i] in keepLabels]
		keepIdxs = np.array(keepIdxs, dtype=np.int)
		startIdxs = startIdxs[keepIdxs]
		endIdxs = endIdxs[keepIdxs]
		labels = labels[keepIdxs]

	# find sections of nearby annotations in the data and group these
	# sections together; we'll concat these groups together to form a ts
	combinedRanges = unionOfRanges(startIdxs, endIdxs, len(X), padLen=padLen)
	rangeGroups = formGroupsOfSize(combinedRanges, groupSize=instancesPerTs,
		shuffle=shuffle)

	# now the hard part--create a LabeledTimeSeries from each of these
	# sections of signal; we have to not only find which annotations
	# fall within each range, but also adjust the start and end indices
	# so that they're correct in the new ts formed by concatenating the
	# data in each range together
	tsList = []
	for groupNum, ranges in enumerate(rangeGroups):

		ranges = sorted(ranges, key=lambda r: r[0]) # sort by range start idx

		dataLenSoFar = 0
		dataInRanges = []
		startsInRanges = []
		endsInRanges = []
		labelsInRanges = []

		for rang in ranges:
			start, end = rang

			firstInRange, lastInRange = whereStartEndPairsInRange(startIdxs,
				endIdxs, start, end)
			idxsInRange = np.arange(firstInRange, lastInRange)

			# move the start and end indices around a bit so that ranges
			# aren't spaced exactly uniformly, which can lead to an
			# artificial semblance of regularity
			if maxPadJitter > 0:
				if firstInRange > 0:
					firstStartIdx = startIdxs[firstInRange]
					prevEndIdx = endIdxs[firstInRange-1]
					gap = firstStartIdx - prevEndIdx
					if gap > 1:
						gap = min(gap - 1, maxPadJitter)
						offset = int(np.random.rand() * gap)
						start -= offset
				if lastInRange < (len(startIdxs) - 1):
					lastEndIdx = endIdxs[lastInRange-1] # last idx not inclusive
					nextStartIdx = startIdxs[lastInRange]
					gap = nextStartIdx - lastEndIdx
					if gap > 1:
						gap = min(gap - 1, maxPadJitter)
						offset = int(np.random.rand() * gap)
						end += offset

			starts = startIdxs[idxsInRange] - start + dataLenSoFar
			ends = endIdxs[idxsInRange] - start + dataLenSoFar
			lbls = labels[idxsInRange]

			startsInRanges += list(starts)
			endsInRanges += list(ends)
			labelsInRanges += list(lbls)

			data = ensure2D(X[start:end])
			dataInRanges.append(data)

			dataLenSoFar += len(data)

		if len(labelsInRanges) < 2: # need more than one pattern instance per ts
			continue

		groupData = np.vstack(dataInRanges)
		groupStarts = np.array(startsInRanges, dtype=np.int)
		groupEnds = np.array(endsInRanges, dtype=np.int)
		groupLabels = np.array(labelsInRanges, dtype=np.object)

		name = "{}-group{}".format(datasetName, groupNum)
		uniqId = hash(name)

		ts = LabeledTimeSeries(groupData, startIdxs=groupStarts,
			endIdxs=groupEnds, labels=groupLabels, name=name, id=uniqId)

		tsList.append(ts)

	return tsList


# ================================================================ Concatenation

def groupDatasetByLabel(X, Y):
	return splitElementsBy(lambda i, x: Y[i], X)


def formGroupsOfSize(collection, groupSize=10, shuffle=False):
	# -note that having |group| = groupSize is not guaranteed;

	if shuffle:
		np.random.shuffle(collection)

	groups = []
	i = 0
	while i < len(collection):
		j = i + groupSize
		groups.append(collection[i:j])
		i += groupSize
	return groups


def concatedTsList(X, Y, instancesPerTs=10, datasetName="Dataset",
	enemyInstancesPerTs=0, **paddingKwargs):
	"""instances -> [LabeledTimeSeries], with each pure wrt class of instances"""
	groupedByClass = groupDatasetByLabel(X, Y)

	# we allow at most one instance of each other class so there's only one
	# repeating "pattern"
	numClasses = len(groupedByClass)
	if enemyInstancesPerTs > numClasses - 1:
		print("concatedTsList(): WARNING: "
			"enemyInstancesPerTs {} > num digits - 1; will be truncated".format(
				enemyInstancesPerTs))
		enemyInstancesPerTs = numClasses - 1

	tsList = []
	for clz, instances in groupedByClass.iteritems():
		groups = formGroupsOfSize(instances, instancesPerTs)

		for groupNum, group in enumerate(groups):

			otherClasses = groupedByClass.keys()
			otherClasses.remove(clz)
			lbls = [clz] * len(group)
			if enemyInstancesPerTs > 0:

				enemyLbls = np.random.choice(otherClasses, enemyInstancesPerTs)
				if enemyInstancesPerTs == 1:
					enemyLbls = [enemyLbls]
				else:
					enemyLbls = list(enemyLbls)
				for dgt in enemyLbls:
					whichRecording = np.random.choice(groupedByClass[dgt])
					group.append(whichRecording)
				allIdxs = np.arange(len(group))
				orderIdxs = np.random.choice(allIdxs, len(allIdxs))
				np.random.shuffle(orderIdxs)

				lbls = lbls + enemyLbls
				lbls = np.array(lbls, dtype=np.object)
				lbls = lbls[orderIdxs]
				groups = [group[idx] for idx in orderIdxs]

			concated, startIdxs, endIdxs = concatWithPadding(
				group, **paddingKwargs)
			name = "{}-class{}-group{}".format(datasetName, clz, groupNum)
			uniqId = hash(name)
			ts = LabeledTimeSeries(data=concated, startIdxs=startIdxs,
				endIdxs=endIdxs, labels=lbls, name=name, id=uniqId)
			tsList.append(ts)

	return tsList

# ================================================================ Data structs


class LabeledTimeSeries(object):

	def __init__(self, data, startIdxs, endIdxs=None, subseqLength=None,
		labels=None, name=None, id=0):
		self.data = ensure2D(data)
		self.startIdxs = np.asarray(startIdxs, dtype=np.int)
		self.labels = np.asarray(labels)
		self.name = name
		self.id = int(id)

		if endIdxs is not None:
			self.endIdxs = np.asarray(endIdxs, dtype=np.int)
			self.subseqLength = None
		elif subseqLength:
			self.endIdxs = self.startIdxs + subseqLength
			self.subseqLength = subseqLength
		else:
			raise ValueError("Either endIdxs or subseqLength must be specified!")

		if labels is None or len(labels) == 0:
			self.labels = np.zeros(len(startIdxs), dtype=np.int) + DEFAULT_LABEL

		if startIdxs is not None and endIdxs is not None:
			# equal lengths
			nStart, nEnd = len(startIdxs), len(endIdxs)
			if nStart != nEnd:
				raise ValueError("Number of start indices must equal number"
					"of end indices! {0} != {1}".format(nStart, nEnd))
			# starts before or equal to ends
			violators = np.where(startIdxs > endIdxs)[0]
			if np.any(violators):
				raise ValueError("Some start indices exceed end indices!"
					"Violators at {}".format(str(violators)))
			# valid indices
			violators = np.where(startIdxs < 0)[0]
			if np.any(violators):
				raise ValueError("Some start indices < 0!"
					"Violators at {}".format(str(violators)))
			violators = np.where(endIdxs > len(data))[0]
			if np.any(violators):
				violatorValues = endIdxs[violators]
				raise ValueError("Some end indices > length of data {}! "
					"Violators {} at {}".format(len(data),
						str(violatorValues), str(violators)))

	def clone(self):
		return LabeledTimeSeries(np.copy(self.data),
			np.copy(self.startIdxs),
			np.copy(self.endIdxs),
			subseqLength=self.subseqLength,
			labels=np.copy(self.labels),
			name=self.name,
			id=self.id
		)

	def plot(self, saveDir=None, capYLim=1000, ax=None, staggerHeights=True,
		yFrac=.9, showBounds=True, showLabels=True, useWhichLabels=None,
		linewidths=2., colors=None, **plotRectKwargs):

		xlimits = [0, len(self.data)]
		ylimits = [self.data.min(), min(capYLim, self.data.max())]
		yMin, yMax = ylimits

		if ax is None:
			plt.figure(figsize=(10, 6))
			ax = plt.gca()

		if not hasattr(linewidths, '__len__'):
			linewidths = np.zeros(self.data.shape[1]) + linewidths

		hasColors = colors is not None and len(colors)
		for i in range(self.data.shape[1]):
			if hasColors:
				ax.plot(self.data[:, i], lw=linewidths[i], color=colors[i])
			else:
				ax.plot(self.data[:, i], lw=linewidths[i])

		ax.set_xlim(xlimits)
		ax.set_ylim(ylimits)
		ax.set_title(self.name)

		hasUseWhichLabels = useWhichLabels is not None and len(useWhichLabels)

		# plot annotations
		if showLabels or showBounds:
			for i in range(len(self.startIdxs)):
				ts, te, label = self.startIdxs[i], self.endIdxs[i], self.labels[i]
				# print "label, useWhichLabels", label,
				if hasUseWhichLabels and label not in useWhichLabels:
					continue

				if showBounds:
					plotRect(ax, ts, te, **plotRectKwargs) # show boundaries

				if showLabels:
					x = ts + (te - ts) / 10
					if staggerHeights: # so labels don't end up on top of one another
						yFrac = .67
						yFrac += .04 * (i // 1 % 2)
						yFrac += .08 * (i // 2 % 2)
						yFrac += .16 * (i // 4 % 2)
					y = yFrac * (yMax - yMin) + yMin # use yFrac passed in if not staggering
					ax.annotate(label, xy=(x, y), xycoords='data')

		if saveDir:
			fileName = self.name + '.pdf'
			if not os.path.exists(saveDir):
				os.makedirs(saveDir)
			path = os.path.join(saveDir, fileName)
			plt.savefig(path)

		return ax

	def plotSubseqs(self, saveDir, **kwargs):
		generateVideos(self.data, dataName=self.name, saveInDir=saveDir,
			rangeStartIdxs=self.startIdxs, rangeEndIdxs=self.endIdxs,
			rangeLabels=self.labels, **kwargs)

# ================================================================ Main

if __name__ == '__main__':
	from doctest import testmod
	testmod()
