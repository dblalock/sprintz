#!/usr/bin/env/python

import numpy as np
from scipy import signal
from scipy.ndimage import filters

# ================================================================
# General purpose funcs
# ================================================================

# ------------------------ create classic patterns

def cylinder(length, amp=1, ampStd=.02, noiseStd=.02):
	"""noisy elevated section, from CBF dataset"""
	amp = amp + np.random.randn(length) * ampStd
	return np.random.randn(length) * noiseStd + amp

def bell(length, amp=1, ampStd=.02, noiseStd=.02):
	"""noisy ascending ramp, from CBF dataset"""
	amp = amp + np.random.randn(1) * ampStd
	return np.linspace(0, amp, length) + np.random.randn(length) * noiseStd

def funnel(length, amp=1, ampStd=.02, noiseStd=.02):
	"""noisy descending ramp, from CBF dataset"""
	return bell(length, amp, ampStd, noiseStd)[::-1]

def sines(length, amp=1, ampStd=.02, noiseStd=.02, periods=1, periodOffset=0):
	amp = amp + np.random.randn() * ampStd
	tstart = 2 * np.pi * periodOffset
	tend = 2 * np.pi * (periods + periodOffset)
	t = np.linspace(tstart, tend, length, endpoint=False)
	return np.sin(t) * amp + np.random.randn(length) * noiseStd

# def warpedSines(length, firstPieceFrac=.33, splitFrac=.5, periods=1, **kwargs):
# 	firstPieceLen = int(firstPieceFrac * length)
# 	secondPieceLen = length - firstPieceLen
# 	firstPiecePeriods = splitFrac * periods
# 	secondPiecePeriods = (1. - splitFrac) * periods
# 	sine1 = sines(firstPieceLen, periods=firstPiecePeriods, **kwargs)
# 	phaseShiftFrac = splitFrac + .5 / secondPieceLen
# 	# phaseShiftFrac = splitFrac
# 	sine2 = sines(secondPieceLen, periods=secondPiecePeriods,
# 		periodOffset=phaseShiftFrac, **kwargs)
# 	return np.r_[sine1, sine2]

def warpedSines(length, origFracs=.5, newFracs=.33, periods=1, **kwargs):
	if not origFracs:
		raise ValueError("Argument origFracs is required; received {}".format(
			origFracs))
	if not newFracs:
		raise ValueError("Argument newFracs is required; received {}".format(
			newFracs))

	# ensure fractions are collections
	if not hasattr(origFracs, '__len__'):
		origFracs = [origFracs]
	if not hasattr(newFracs, '__len__'):
		newFracs = [newFracs]

	# ensure fractions given are between 0 and 1
	if np.min(origFracs) < 0.:
		raise ValueError("origFracs contained values < 0!")
	if np.max(origFracs) > 1.:
		raise ValueError("origFracs contained values > 1!")
	if np.min(newFracs) < 0.:
		raise ValueError("newFracs contained values < 0!")
	if np.max(newFracs) > 1.:
		raise ValueError("newFracs contained values > 1!")

	# have each start with 0, end with 1, and be monotonic and nonrepeated
	origFracs = [0] + sorted(origFracs) + [1.]
	newFracs = [0] + sorted(newFracs) + [1.]
	origFracs = np.unique(origFracs)
	newFracs = np.unique(newFracs)

	if len(origFracs) != len(newFracs):
		raise IndexError("origFracs length {} != newFracs length {}".format(
			len(origFracs), len(newFracs)))

	print "origFracs", origFracs
	print "newFracs", newFracs

	pieces = []
	numPieces = len(origFracs) - 1
	# for i, origFrac in enumerate(origFracs[:-1]): # :-1 since we appended a 1
	for i in range(numPieces): # :-1 since we appended a 1
		origFrac = origFracs[i]
		newFrac = newFracs[i]
		# determine end of this piece
		# if isLastPiece: # this is last piece
		# 	nextOrigFrac = 1.
		# 	nextNewFrac = 1.
		# else:
		# 	nextOrigFrac = origFracs[i+1]
		# 	nextNewFrac = newFracs[i+1]

		origFrac = origFracs[i]
		newFrac = newFracs[i]
		nextOrigFrac = origFracs[i+1]
		nextNewFrac = newFracs[i+1]

		deltaOrigFrac = nextOrigFrac - origFrac
		deltaNewFrac = nextNewFrac - newFrac

		isLastPiece = i == (numPieces - 1)
		if isLastPiece: # ensure output is correct length (despite rounding)
			pieceLen = length - sum([len(piece) for piece in pieces])
		else:
			pieceLen = int(deltaNewFrac * length)
		print "creating piece of len", pieceLen
		piecePeriods = deltaOrigFrac * periods
		piecePeriodOffset = origFrac
		sinewave = sines(pieceLen, periods=piecePeriods,
			periodOffset=piecePeriodOffset, **kwargs)
		pieces.append(sinewave)

	return np.hstack(pieces)

def randconst(shape, mean=0., std=1.):
	try:
		return np.random.randn(*shape) * std + mean
	except:
		return np.random.randn(shape) * std + mean

def randwalk(shape, std=1):
	try:
		if len(shape) > 1 and not any([dim == 1 for dim in shape]): # specify axis=1 if 2D
			return np.cumsum(randconst(shape, std=std), axis=1)
		return np.cumsum(randconst(shape, std=std))
	except:
		return np.cumsum(randconst(shape, std=std))

def notSoRandomWalk(shape, std=1, trendFilterLength=32, lpfLength=16):
	"""bandpass filter a random walk so that the low-frequency trend /
	drift is eliminated and the high-frequency noise is attenuated"""
	walk = randwalk(shape, std=std)
	filt = np.hamming(trendFilterLength)
	filt /= np.sum(filt)
	whichAxis = len(walk.shape) > 1 # 0 iff 1d, else 1
	# subtract baseline drift, roughly
	trend = filters.convolve1d(walk, weights=filt, axis=whichAxis, mode='reflect')
	walk -= trend
	# subtract noisey spikes
	walk = filters.convolve1d(walk, weights=np.hamming(lpfLength), axis=whichAxis, mode='reflect')
	return walk

def randWithFreqMagMatching(X, shape=None):
	"""Given a data matrix X, returns a matrix of the same dimensions
	whose rows have the same frequency magnitude spectra as the rows of
	X, but randomly shuffled phases.
	"""
	if shape is None:
		shape = X.shape
	whichAxis = len(shape) > 1 # 1D -> 0; 2D -> 1
	avgFFT = np.fft.fft(X, axis=whichAxis)
	mags = np.absolute(avgFFT)
	phases = np.angle(avgFFT)
	if whichAxis:
		for i in range(len(X)):
			np.random.shuffle(phases[i]) # shuffle phase of each row in place
	else:
		np.random.shuffle(phases)
	noiseFFT = mags * np.exp(1.j * phases)
	noise = np.fft.ifft(noiseFFT)
	return noise

# ------------------------ utility funcs

def embedSubseq(fullseq, subseq, startIdx=None, sameMean=True):
	if startIdx is None:
		maxStartIdx = len(fullseq) - len(subseq)
		startIdx = np.random.choice(np.arange(maxStartIdx + 1))
	endIdx = startIdx+len(subseq)
	mean = np.mean(fullseq[startIdx:endIdx])
	# print "embedSubseq(), ", len(subseq), startIdx, endIdx, mean*sameMean
	fullseq[startIdx:endIdx] = subseq + mean*sameMean

def createMotif(background, instance1, instance2, sameMean=False,
	returnStartIdxs=False, **sink):

	maxIdx1 = len(background)/2 - len(instance1) # can't encroach on 2nd half
	start1 = np.random.choice(np.arange(maxIdx1))
	maxEndIdx2 = (len(background) - len(instance2)) - len(background)/2 # can't encroach on 1st half
	start2 = np.random.choice(np.arange(maxEndIdx2)) + len(background)/2
	seq = background.copy()
	embedSubseq(seq, instance1, start1, sameMean)
	embedSubseq(seq, instance2, start2, sameMean)
	if returnStartIdxs:
		return seq, start1, start2
	return seq

def seedRng(seed):
	if seed:
		np.random.seed(seed)

def randWarpingPath(seqLength, stepConstraints=True, reallyWarped=False):
	# step constraints = at most one in each direction

	maxIdx = seqLength - 1
	i = 0
	j = 0
	wasHorz = False
	wasVert = False
	path = [(0,0)]

	# random choices--equiprobable by default
	horzThresh = .33
	vertThresh = .67

	# actively give it a weird warping path by just increasing i for a while
	if reallyWarped:
		for k in range(1, int(maxIdx / 2)):
			i = k
			j = int(k / 4)
			path.append((i, j))

		horzThresh = .5
		vertThresh = .75

	canIncrement_i = i < maxIdx
	canIncrement_j = j < maxIdx
	while canIncrement_i or canIncrement_j:
		randNum = np.random.rand()
		if (not canIncrement_i) or (canIncrement_j and randNum < horzThresh and not wasHorz):
			# horizontal step
			j += 1
			wasHorz = True and stepConstraints
			wasVert = False
		elif (not canIncrement_j) or (canIncrement_i and randNum < vertThresh and not wasVert):
			# vertical step
			i += 1
			wasHorz = False
			wasVert = True and stepConstraints
		elif canIncrement_i and canIncrement_j:
			# diagonal step
			i += 1
			j += 1
			wasHorz = False
			wasVert = False

		path.append((i,j))
		canIncrement_i = i < maxIdx
		canIncrement_j = j < maxIdx

	return path

def warpedSeq(seq, sameLength=True, useI=True, **kwargs):
	path = randWarpingPath(len(seq), **kwargs) # list of (i,j) pairs
	idxs_i, idxs_j = zip(*path) # tuple of i vals, tuple of j vals
	idxs = idxs_i if useI else idxs_j # use i idxs or j idxs
	warped = seq[np.asarray(idxs)]
	if sameLength:
		warped = signal.resample(warped, len(seq))
	return warped

def appendZeros(A, length, axis=1):
	if length < 1:
		return A
	A = np.asarray(A)
	if len(A.shape) == 1:
		return np.r_[A, np.zeros(length)]
	if axis == 0:
		return np.vstack((A, np.zeros((length, A.shape[1]))))
	if axis == 1:
		return np.hstack((A, np.zeros((A.shape[0], length))))


def ensure2D(X): # TODO if we add deps, use impl of this in arrays.py
	X = np.asarray(X)
	if len(X.shape) == 1:
		X = X.reshape((-1, 1)) # ensure 2d array
	return X


def ensureIterable(seqs): # TODO deprecated; use ensureIsCollection
	if not isinstance(seqs, (set, frozenset, list, tuple)):
		seqs = [seqs]
	return seqs


def ensureIsCollection(seqs):
	if not isinstance(seqs, (set, frozenset, list, tuple)):
		seqs = [seqs]
	return seqs


def addNoiseDims(X, numToAdd=-1, noiseType='randwalk'):
	X = ensure2D(X)

	if numToAdd == 0:
		return X
	if numToAdd < 0:
		numToAdd = X.shape[1]
	elif numToAdd < 1.:
		numToAdd = int(X.shape[1] * numToAdd)

	noiseShape = (X.shape[0], numToAdd)
	if noiseType is None or noiseType == 'randwalk':
		noise = randwalk(noiseShape)
	elif noiseType == 'white' or noiseType == 'gaussian':
		noise = randconst(noiseShape)
	else:
		raise ValueError("Unrecognized noise type: {}".format(noiseType))

	return np.hstack((X, ensure2D(noise)))

def addAdversarialDims(X, numToAdd, startIdxs, endIdxs):
	X = ensure2D(X)

	if numToAdd == 0:
		return X
	if numToAdd < 0:
		numToAdd = X.shape[1]
	elif numToAdd < 1.:
		numToAdd = int(X.shape[1] * numToAdd)

	signalLength = len(X)
	newDims = []
	for i in range(numToAdd):
		sig = createAdversarialSignal(startIdxs, endIdxs, signalLength)
		newDims.append(sig.reshape((-1,1))) # col vect

	newDims = np.hstack(newDims)
	return np.hstack((X, newDims))

def createAdversarialSignal(startIdxs, endIdxs, signalLength):
	assert(len(startIdxs) == len(endIdxs))
	numInstances = len(startIdxs)
	# avgLength =

	pairs = np.arange(numInstances)
	if len(pairs) % 2 != 0: # handle odd numbers of instances
		pairs = np.append(pairs, np.random.choice(pairs)) # duplicate some idx
	np.random.shuffle(pairs)
	pairs = pairs.reshape((-1, 2)) # random

	out = np.zeros(signalLength)
	for pair in pairs:
		start1, end1 = startIdxs[pair[0]], endIdxs[pair[0]]
		start2, end2 = startIdxs[pair[1]], endIdxs[pair[1]]
		length1 = end1 - start1 # end idxs not inclusive
		length2 = end2 - start2
		subseq = randwalk(length1)
		negSeq = -subseq
		if length1 != length2:
			negSeq = signal.resample(negSeq, length2)
		out[start1:end1] = subseq
		out[start2:end2] = negSeq

	return out


# ================================================================
# Create particular synthetic datasets (for prototyping / smoke testing)
# ================================================================

DEFAULT_MOTIF_LEN = 400
DEFAULT_INSTANCE_LEN = 50

# ------------------------------------------------
# Single time series
# ------------------------------------------------

def trianglesMotif(noise=.02, backgroundNoise=.02, seed=None, **kwargs):
	seedRng(seed)
	background = randconst(DEFAULT_MOTIF_LEN, std=backgroundNoise)
	m = DEFAULT_INSTANCE_LEN
	inst1 = funnel(m, ampStd=0, noiseStd=noise)
	inst2 = funnel(m, ampStd=0, noiseStd=noise)
	return createMotif(background, inst1, inst2, **kwargs), m

def rectsMotif(noise=.02, backgroundNoise=.02, seed=None, **kwargs):
	seedRng(seed)
	background = randconst(DEFAULT_MOTIF_LEN, std=backgroundNoise)
	m = DEFAULT_INSTANCE_LEN
	inst1 = cylinder(m, ampStd=0, noiseStd=noise)
	inst2 = cylinder(m, ampStd=0, noiseStd=noise)
	return createMotif(background, inst1, inst2, sameMean=False, **kwargs), m

def sinesMotif(noise=0, backgroundNoise=.02, periods=1, seed=None, **kwargs):
	seedRng(seed)
	background = randconst(DEFAULT_MOTIF_LEN, std=backgroundNoise)
	m = DEFAULT_INSTANCE_LEN
	inst1 = sines(m, ampStd=0, noiseStd=noise, periods=periods)
	inst2 = sines(m, ampStd=0, noiseStd=noise, periods=periods)
	return createMotif(background, inst1, inst2, sameMean=False, **kwargs), m

def multiShapesMotif(noise=0, backgroundNoise=.02, periods=1, seed=None, **kwargs):
	seedRng(seed)
	background = randconst((DEFAULT_MOTIF_LEN,3)) * backgroundNoise
	m = DEFAULT_INSTANCE_LEN
	inst1 = np.c_[funnel(m, ampStd=0, noiseStd=noise),
		sines(m, ampStd=0, noiseStd=noise, periods=periods),
		bell(m, ampStd=0, noiseStd=noise)]
	inst2 = np.c_[funnel(m, ampStd=0, noiseStd=noise),
		sines(m, ampStd=0, noiseStd=noise, periods=periods),
		bell(m, ampStd=0, noiseStd=noise)]
	return createMotif(background, inst1, inst2, sameMean=False, **kwargs), m

def makeThreeTriangles(length=400, m=40, noise=0, returnStartIdxs=False):
	patterns = [funnel(m, noiseStd=noise) for i in range(3)]
	startIdxs = [60, 160, 330]
	seq = randconst(length, std=.05) # noise std dev = .05

	for idx, p in zip(startIdxs, patterns):
		embedSubseq(seq, p, idx)

	return seq if returnStartIdxs else seq, startIdxs

def makeTwoTriangles(length=400, m=40, noise=0, returnStartIdxs=False):
	if hasattr(m, '__len__'):
		patterns = [funnel(mi, noiseStd=noise) for mi in m]
	else:
		patterns = [funnel(m, noiseStd=noise) for i in range(2)]
	startIdxs = [110, 280]
	seq = randconst(length, std=.05) # noise std dev = .05

	for idx, p in zip(startIdxs, patterns):
		embedSubseq(seq, p, idx)

	return seq if returnStartIdxs else seq, startIdxs

# ------------------------------------------------
# Multiple time series
# ------------------------------------------------

def collectionOfTsUsingCreationFunc(func, count, **kwargs):
	return [func(**kwargs) for i in range(count)]

def alignSequentialEndpoints(seqs):
	"""add values to each seq in seqs such that seqs[i, -1] = seqs[i+1, 0]"""
	aligned = [seqs[0]]
	prevSeq = seqs[0]
	# prevSeq = ensure2D(seqs[0])
	for seq in seqs[1:]:
		# seq = ensure2D(seq)
		gap = prevSeq[-1] - seq[0]
		adjustedSeq = seq + gap
		prevSeq = adjustedSeq
		aligned.append(adjustedSeq)

	return aligned

def concatSeqs(seqs, axis=0):
	asArrays = [np.asarray(seq) for seq in seqs]
	numDims = [len(seq.shape) for seq in asArrays]
	assert(len(np.unique(numDims)) == 1) # must all have same dimensionality

	if numDims[0] == 1 or axis == 1:
		return np.hstack(asArrays)
	elif numDims[0] == 2 and axis == 0:
		return np.vstack(asArrays)
	else:
		raise ValueError("Does not support ndarrays with n > 2")

def concatWithAlignedEndpoints(seqs):
	alignedSeqs = alignSequentialEndpoints(seqs)
	return concatSeqs(alignedSeqs)

def createPadding(seqs, minPaddingFractionOfLength=1., maxPaddingFractionOfLength=1.,
	padFunc=randwalk, paddingStdDevRatio=1., **kwargs):
	"""creates n+1 padding seqs to go around the n seqs based on their lengths"""
	# ensure seqs is a collection so we can iterate thru it
	# wasCollection = True
	seqs = ensureIterable(seqs)

	# determine shape of padding for each seq
	seqLengths = np.array([len(seq) for seq in seqs])
	minLengths = minPaddingFractionOfLength * seqLengths
	maxLengths = maxPaddingFractionOfLength * seqLengths
	maxLengths = np.maximum(minLengths, maxLengths)
	lengthDiffs = maxLengths - minLengths
	padLengths = np.random.rand(len(seqs)) * lengthDiffs + minLengths
	padLengths = padLengths.astype(np.int)
	if len(seqs[0].shape) > 1:
		nDims = seqs[0].shape[1]
		padShapes = [[padLen, nDims] for padLen in padLengths]
		padShapes.append(padShapes[-1])
		padShapes = np.array(padShapes)
	else:
		padShapes = np.append(padLengths, padLengths[-1]) # have padding after end

	# create padding; should have variance that's the specified ratio of the
	# original data's variance so that the shape of the padding is meaningful
	padding = [padFunc(shape=padShape, **kwargs) for padShape in padShapes]
	if paddingStdDevRatio > 0.:
		seqStd = np.mean([np.std(seq) for seq in seqs])
		paddingStd = np.mean([np.std(pad) for pad in padding])
		currentRatio = paddingStd / seqStd
		multiplyBy = paddingStdDevRatio / currentRatio
		padding = [pad * multiplyBy for pad in padding]

	return padding

def concatWithPadding(seqs, **paddingKwargs):
	seqs = ensureIterable(seqs)

	seqs = [ensure2D(seq) for seq in seqs]
	padding = createPadding(seqs, **paddingKwargs)
	padding = [ensure2D(pad) for pad in padding]

	totalSeqLength = np.sum([len(seq) for seq in seqs])
	totalPadLength = np.sum([len(pad) for pad in padding])
	totalLen = totalSeqLength + totalPadLength
	nDims = seqs[0].shape[1]

	padded = np.empty((totalLen, nDims))
	seqStartIdxs = np.empty(len(seqs), dtype=np.int)
	seqEndIdxs = np.empty(len(seqs), dtype=np.int)
	currentIdx = 0
	prevValue = 0.
	for i, seq in enumerate(seqs):
		seqLen = len(seq)
		pad = padding[i]
		padLen = len(pad)

		pad += (prevValue - pad[0]) # force equal endpoints
		prevValue = pad[-1]
		padded[currentIdx:(currentIdx+padLen)] = pad
		currentIdx += padLen

		seq += (prevValue - seq[0])
		prevValue = seq[-1]
		padded[currentIdx:(currentIdx+seqLen)] = seq
		seqStartIdxs[i] = currentIdx
		currentIdx += seqLen
		seqEndIdxs[i] = currentIdx
	finalPad = padding[-1]
	padded[currentIdx:] = finalPad + (prevValue - finalPad[0])

	return padded, seqStartIdxs, seqEndIdxs

def makeWhiteNoiseSeqs(count=10, shape=50, **kwargs):
	return collectionOfTsUsingCreationFunc(randconst, count, shape=shape, **kwargs)

def makeRandWalkSeqs(count=10, shape=50, **kwargs):
	return collectionOfTsUsingCreationFunc(randwalk, count, shape=shape, **kwargs)

def makeTriangleSeqs(count=2, shape=50, **kwargs):
	return collectionOfTsUsingCreationFunc(funnel, count, length=shape, **kwargs)

def makeSinesSeqs(count=2, shape=50, **kwargs):
	return collectionOfTsUsingCreationFunc(sines, count, length=shape, **kwargs)

def makeSinesDataset(numSines=2, numNoise=10, startIdx=60, warped=False, **kwargs):
	# sines = makeSinesSeqs(numSines, shape=80, noiseStd=0.0, **kwargs)
	sines = makeSinesSeqs(numSines, shape=80, noiseStd=0.1, **kwargs)
	if warped:
		sines = map(lambda s: warpedSeq(s), sines)
	background = makeRandWalkSeqs(numNoise, shape=200, std=.5, **kwargs)
	for i, s in enumerate(sines):
		embedSubseq(background[i], s, startIdx)

	return background

# ================================================================
# Testing
# ================================================================

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	# for i in range(5):
	# plt.plot(cylinder(20), 'g')
	# plt.plot(bell(20), 'b')
	# plt.plot(funnel(20), 'k')
	# plt.plot(sines(20), 'r')
	# plt.plot(createMotif(randwalk(200, std=.1), bell(30), bell(32)))
	# plt.plot(createMotif(randwalk(200, std=.1), sines(30), sines(32)))
	# plt.plot(trianglesMotif()[0], lw=2)
	# plt.plot(rectsMotif()[0], lw=2)
	plt.plot(sinesMotif()[0], lw=2)
	plt.show()
