#!/bin/env/python

import numpy as np
from scipy import signal
from scipy import stats

DEFAULT_NONZERO_THRESH = .001

# ------------------------------- Array attribute checks

def isScalar(x):
	return not hasattr(x, "__len__")


def isSingleton(x):
	return (not isScalar(x)) and (len(x) == 1)


def is1D(x):
	return len(x.shape) == 1


def is2D(x):
	return len(x.shape) == 2


def nrows(A):
	return A.shape[0]


def ncols(A):
	return A.shape[1]


# ------------------------------- Array manipulations

def ensure2D(X):
	X = np.asarray(X)
	if len(X.shape) == 1:
		X = X.reshape((-1, 1)) # ensure 2d array
	return X


def asColumnVect(V):
	return V.reshape((V.size,1))


def colsAsList(A):
	if len(A.shape) < 2:
		return A
	return np.hsplit(A, A.shape[1])


def addZeroCols(A, howMany=1, prepend=False):
	N, P = A.shape
	padding = np.zeros((N, howMany))
	if prepend:
		return np.hstack((padding, A))
	return np.hstack((A, padding))


def addZeroRows(A, howMany=1, prepend=False):
	N, P = A.shape
	padding = np.zeros((howMany, P))
	if prepend:
		return np.vstack((padding, A))
	return np.vstack((A, padding))


def centerInMatOfSize(A, numRows=-1, numCols=-1, fill=0):
	was1D = len(A.shape) == 1
	A = ensure2D(A)
	numRows = max(A.shape[0], numRows)
	numCols = max(A.shape[1], numCols)

	if numRows == A.shape[0] and numCols == A.shape[1]:
		return A

	mat = np.zeros((numRows, numCols)) + fill

	rowStartIdx = (numRows - A.shape[0]) // 2
	colStartIdx = (numCols - A.shape[1]) // 2
	rowEndIdx = rowStartIdx + A.shape[0]
	colEndIdx = colStartIdx + A.shape[1]

	mat[rowStartIdx:rowEndIdx, colStartIdx:colEndIdx] = A

	if was1D:
		mat = mat.ravel()
	return mat


def prependOnesCol(A):
	N, P = A.shape
	return np.hstack((np.ones((N,1)), A))


def prependOnesRow(A):
	N, P = A.shape
	return np.vstack((np.ones((1,P)), A))


def nonzeroRows(A, thresh=DEFAULT_NONZERO_THRESH):
	return np.where(np.sum(np.abs(A), axis=1) > thresh)[0]


def nonflatRows(A):
	return nonzeroRows(meanNormalizeRows(A))


def nonzeroCols(A, thresh=DEFAULT_NONZERO_THRESH):
	"""Return the columns of A that contain nonzero elements"""
	if np.any(np.isnan(A)):
		print("WARNING: nonzeroCols: cols containing NaNs may be removed")
	if is2D(A):
		return np.where(np.sum(np.abs(A), axis=0) > thresh)[0]  # [0] to unpack singleton
	return np.where(np.abs(A))[0]


def removeCols(A, cols):
	if cols is not None:
		return np.delete(A, cols, 1)
	return A


def removeZeroCols(A, thresh=DEFAULT_NONZERO_THRESH):
	return A[:, nonzeroCols(A, thresh=thresh)]


def removeZeroRows(A, thresh=DEFAULT_NONZERO_THRESH):
	return A[nonzeroRows(A, thresh=thresh)]


def removeFlatRows(A):
	return A[nonflatRows(A), :]


def extractCols(A, cols):
	extracted = A[:, cols]
	remnants = removeCols(A, cols)
	return extracted, remnants


def L2NormalizeRows(A):
	return A / np.linalg.norm(A, axis=1).reshape((-1, 1))


def L2NormalizeCols(A):
	return A / np.linalg.norm(A, axis=1)


def meanNormalizeRows(A):
	rowMeans = np.mean(A, 1).reshape(A.shape[0], 1)
	return A - rowMeans


def meanNormalizeCols(A):
	return A - np.mean(A, 0)


def stdNormalizeCols(A, removeZeros=False, thresh=DEFAULT_NONZERO_THRESH):
	if removeZeros:
		A = removeZeroCols(A, thresh=thresh)
	# else:
	# 	nonzeros = nonzeroCols(A)
	# 	A[:, nonzeros] = A[:,nonzeroCols[A]]
	colStds = np.std(A, axis=0)
	colStds[colStds < DEFAULT_NONZERO_THRESH] = 1 # ignore zeros
	return A / colStds


def stdNormalizeRows(A, removeZeros=False, thresh=DEFAULT_NONZERO_THRESH):
	if removeZeros:
		A = removeZeroRows(A, thresh=thresh)
	rowStds = np.std(A, axis=1).reshape((-1,1))
	rowStds[rowStds < DEFAULT_NONZERO_THRESH] = 1 # ignore zeros
	return A / rowStds


def zNormalize(v):
	return (v - v.mean()) / v.std()


def zNormalizeCols(A, removeZeros=False):
	if len(A.shape) == 1:
		return zNormalize(A)
	return stdNormalizeCols(meanNormalizeCols(A), removeZeros)


def zNormalizeRows(A, removeZeros=False):
	if len(A.shape) == 1:
		return zNormalize(A)
	return stdNormalizeRows(meanNormalizeRows(A), removeZeros)


def meanNormalizeEachDim(X, nDims):
	Xnorm = np.empty(X.shape)
	for i, subseq in enumerate(X):
		s = subseq.reshape((nDims, -1))
		s = meanNormalizeCols(s)
		Xnorm[i] = s.flatten()
	return Xnorm


def zNormalizeEachDim(X, nDims, removeZeros=False):
	Xnorm = np.empty(X.shape)
	for i, subseq in enumerate(X):
		s = subseq.reshape((nDims, -1))
		s = zNormalizeCols(s, removeZeros=False)
		# Xnorm[i] = s.flatten()
		Xnorm[i] = s.T.flatten() # stack cols (data from each dim) sequentially
	return Xnorm


def augmentRowsToNormOne(X, targetNorm=1.):
	# augment such that norm of each row is 1
	newNorms = np.linalg.norm(X, axis=1).reshape((-1,1))
	return np.hstack((X, np.sqrt(targetNorm - newNorms**2)))


def augmentVectToFixedNorm(v, targetNorm=1.):
	v = v.flatten()
	norm = np.linalg.norm(v)
	return np.r_[v, np.sqrt(targetNorm - norm*norm)]


def mipsNormalizeRows(X, maxMagnitude=1., data=False, query=False):
	"""
	Normalizes rows of X to such that the largest row has norm maxMagnitude,
	and then augments each row with a value such that its magnitude is 1.
	maxMagnitude *must* be <= 1.

	If data and query are both false, each x_i is augmented
	with sqrt(1-||x_i||^2). If data=True, each x_i is augmented with
	sqrt(1-||x_i||^2), 0. If query=True, each x_i is augmented with
	0, sqrt(1-||x_i||^2). data and query must not both be true.
	"""
	if data and query:
		raise ValueError("Data cannot be normalized as both data and query")

	if maxMagnitude > 1.:
		raise ValueError("Need 0. < maxMagnitude <= 1. Got {}".format(maxMagnitude))

	norms = np.linalg.norm(X, axis=1)
	maxNorm = np.max(norms)
	if maxNorm > 1.:
		whereBad = np.where(norms > 1.)[0]
		raise ValueError("Rows had norm > 1!: {}".format(str(whereBad)))

	divideRowsBy = np.zeros((-1, 1)) + maxNorm
	X /= divideRowsBy
	X *= maxMagnitude

	if (not data) and (not query): 	# append sqrt(1-||x_i||^2)
		return augmentRowsToNormOne(X)
	elif data: 						# append sqrt(1-||x_i||^2), 0
		X = augmentRowsToNormOne(X)
		return addZeroCols(X, 1)
	elif query: 					# append 0, sqrt(1-||x_i||^2)
		X = addZeroCols(X, 1)
		return augmentRowsToNormOne(X)


def normalizeMat(X, origNumDims, how=None, maxMagnitude=1.):
	"""
	how='each': zNormalize each dimension in each subseq
	how='all': zNormalize concatenation of all dims
	how='each_mean': mean-normalize each dimension in each subseq
	how='all_mean': mean-normalize concatenation of all dims
	how='l2': scale each row of X so that it has L2 norm 1
	how='mips': divide each row by the maximum norm of any row and
		augment each row with one extra entry such that the norm of
		each row becomes 1.
	else: return X

	maxMagnitude: in the case of MIPS normalization, scales down each
	vector (before augmentation) so that the largest vector has norm
	of maxMagnitude. The final magnitude of each row is still 1.
	"""
	# TODO all vs each for L1, L2 norms

	how = how.lower()
	if how == 'each':
		return zNormalizeEachDim(X, origNumDims)
	elif how == 'all':
		return zNormalizeRows(X)
	elif how == 'each_mean':
		return meanNormalizeEachDim(X, origNumDims)
	elif how == 'all_mean':
		return meanNormalizeRows(X)
	elif how == 'mips':
		return mipsNormalizeRows(X, maxMagnitude)
	elif how == 'mips_data':
		return mipsNormalizeRows(X, maxMagnitude, data=True)
	elif how == 'mips_query':
		return mipsNormalizeRows(X, maxMagnitude, query=True)
	elif how == 'l2' or how == 'euclidean':
		return L2NormalizeRows(X)
	else:
		return X


def normalizeCols(A):
	return normalizeRows(A.T).T


def normalizeRows(A):
	A = meanNormalizeRows(A)
	rowIdxs = nonzeroRows(A)
	norms = np.linalg.norm(A[rowIdxs], axis=1)
	A[rowIdxs] /= norms.reshape((len(norms), 1))
	return A


def cosineSim(x, y):
	x -= np.mean(x)
	y -= np.mean(y)
	x /= np.linalg.norm(x)
	y /= np.linalg.norm(y)
	return np.dot(x, y)


def computeSlope(y):
	"""compute slope of y best fit line to y under a linear regression"""
	x = np.arange(len(y))
	y -= np.mean(y)
	slope, _, _, _, _ = stats.linregress(x, y)
	return slope


def detrend(y):
	"""subtracts off the trend line fit by a linear regression"""
	x = np.arange(len(y))
	y -= np.mean(y)
	slope = computeSlope(y)
	y -= x * slope
	return y


def downsampleMat(A, rowsBy=1, colsBy=1):
	if len(A.shape) == 1:
		return signal.decimate(A, rowsBy, n=(rowsBy-1))
	if rowsBy != 1:
		A = signal.decimate(A, rowsBy, n=(rowsBy-1), axis=0)
	if colsBy != 1:
		A = signal.decimate(A, rowsBy, n=(colsBy-1), axis=1)
	return A
		# A = A.reshape(-1, 1)
		# newLen = int(A.shape[0] / float(rowsBy))
		# resampled = imresize(A, (newLen, 1))
		# return resampled.flatten()
	# newShape = A.shape / np.array([rowsBy, colsBy], dtype=np.float)
	# newShape = newShape.astype(np.int)  # round to int
	# return imresize(A, newShape)


def zeroOneScaleMat(A):
	minVal = np.min(A)
	maxVal = np.max(A)
	return (A - minVal) / (maxVal - minVal)


def array2tuple(V):
	return tuple(map(tuple,V))[0]


def dictsTo2DArray(dicts):
	"""returns a dense 2D array with a column for each key in any dictionary
	and a row for each dictionary; where a key is absent in a dictionary, the
	corresponding entry is populated by a 0. Note that the columns are
	ordered according to the sorting of the keys

	EDIT: also returns the column headers as a tuple
	"""
	allKeys = set()
	for d in dicts:
		allKeys.update(d.keys())
	sortedKeys = sorted(allKeys)
	numKeys = len(sortedKeys)
	numDicts = len(dicts)
	keyIdxs = np.arange(numKeys)
	key2Idx = dict(zip(sortedKeys, keyIdxs))
	ar = np.zeros((numDicts, numKeys))
	for i, d in enumerate(dicts):
		for key, val in d.iteritems():
			idx = key2Idx[key]
			ar[i, idx] = val
	return ar, tuple(sortedKeys)


# ------------------------------- Array searching

def findRow(A, q):
	"""return the row indices of all rows in A that match the vector q"""
	assert(ncols(A) == len(q))
	assert(is1D(q))
	rowEqualsQ = np.all(A == q, axis=1)
	return np.where(rowEqualsQ)


def numNonZeroElements(A):
	return len(np.where(A.flatten()))


def idxsOfRelativeExtrema(x, maxima=True, allowEq=False, axis=0):
	"""
	>>> idxsOfRelativeExtrema([2,1,5])
	array([0, 2])
	>>> idxsOfRelativeExtrema([2,1])
	array([0])
	>>> idxsOfRelativeExtrema([1,2])
	array([1])
	>>> idxsOfRelativeExtrema([2,1,5], maxima=False)
	array([1])
	>>> idxsOfRelativeExtrema([1,1,1], allowEq=False)
	array([], dtype=int64)
	>>> idxsOfRelativeExtrema([0,1,1,1,0], allowEq=False)
	array([], dtype=int64)
	>>> idxsOfRelativeExtrema([1,1,1], allowEq=True)
	array([0, 1, 2])
	"""
	if len(x) == 0:
		return np.empty(1) # []
	if len(x) == 1:
		return np.zeros(1) # [0]

	x = np.asarray(x)
	pad = -np.inf if maxima else np.inf
	if maxima:
		if allowEq:
			func = np.greater_equal
		else:
			func = np.greater
	else:
		if allowEq:
			func = np.less_equal
		else:
			func = np.less
	if len(x.shape) == 1:
		x = np.r_[x, pad] # combine with wrap to check endpoints
		return signal.argrelextrema(x, func, mode='wrap')[0]
	elif axis == 0:
		pad = np.zeros((1, x.shape[1])) + pad
		x = np.vstack((x, pad))
		return signal.argrelextrema(x, func, mode='wrap', axis=axis)
	elif axis == 1:
		pad = np.zeros((x.shape[0], 1)) + pad
		x = np.hstack((x, pad))
		return signal.argrelextrema(x, func, mode='wrap', axis=axis)
	else:
		raise NotImplementedError("only supports axis={0, 1}!")


def slidingMinimaIdxs(v, windowLen, pastBeginning=False, pastEnd=False):
	"""
	Parameters
	----------
	v : array_like, 1D
		The input data
	windowLen : int > 0
	pastBeginning : bool, optional
		True - sliding window positions begin as soon as the window intersects
		with the first element
		False - sliding window positions begin when the window is full
	pastEnd : bool, optional
		True - sliding window positions continue as long as the window
		intersects with at least one element
		False - sliding window positions end when the window would no longer
		be full

	Note that if only one of pastBeginning and pastEnd is True, the returned
	array will have the same length as the input array

	Returns
	-------
	idxs : array, 1D
		An array such that idxs[i] is the index of the smallest value in v
		that's included when a sliding window is at position i.


	>>> v = np.array([0, 0, -1, 1, 1])
	>>> idxs = slidingMinimaIdxs(v, 2)
	>>> v[idxs]
	array([ 0, -1, -1,  1])
	>>>
	>>> idxs = slidingMinimaIdxs(v, 2, pastBeginning=True)
	>>> v[idxs]
	array([ 0,  0, -1, -1,  1])
	>>>
	>>> idxs = slidingMinimaIdxs(v, 2, pastEnd=True)
	>>> v[idxs]
	array([ 0, -1, -1,  1,  1])
	>>>
	>>> v = np.array([0, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 5,-2,-3,-2,-1, 0])
	>>> idxs = slidingMinimaIdxs(v, 1)
	>>> v[idxs]
	array([ 0,  0,  0,  1,  2,  3,  4,  3,  2,  1,  0,  5, -2, -3, -2, -1,  0])
	>>>
	>>> idxs = slidingMinimaIdxs(v, 2)
	>>> v[idxs]
	array([ 0,  0,  0,  1,  2,  3,  3,  2,  1,  0,  0, -2, -3, -3, -2, -1])
	>>>
	>>> idxs = slidingMinimaIdxs(v, 3)
	>>> v[idxs]
	array([ 0,  0,  0,  1,  2,  3,  2,  1,  0,  0, -2, -3, -3, -3, -2])
	>>>
	>>> idxs = slidingMinimaIdxs(v, 3, pastEnd=True)
	>>> v[idxs]
	array([ 0,  0,  0,  1,  2,  3,  2,  1,  0,  0, -2, -3, -3, -3, -2, -1,  0])
	"""

	if windowLen < 1:
		raise ValueError("Invalid windowLen {}".format(windowLen))
	if windowLen == 1:
		return np.arange(len(v))

	if pastEnd:
		maxVal = np.max(v)
		fill = np.zeros(windowLen - 1) + maxVal + 1 # never a new minimum
		v = np.append(v, fill)

	n = len(v)
	m = windowLen

	idxs = np.empty(n, dtype=np.int)
	idxs[0] = 0

	# could use stack of size m, but this avoids boundary conditions
	stackDeaths = np.empty(n, dtype=np.int)
	stackVals = np.empty(n)

	stackDeaths[0] = m
	stackVals[0] = v[0]
	first = 0
	last = 0

	for i in range(1, n):
		if i == stackDeaths[first]: # best value expired
			first += 1
		newVal = v[i]
		while stackVals[last] >= newVal and last >= first:
			last -= 1
		last += 1
		stackDeaths[last] = i + m
		stackVals[last] = newVal
		idxs[i] = stackDeaths[first] - m

	if not pastBeginning: # only return idxs where window was full
		return idxs[m-1:]

	return idxs


def slidingMaximaIdxs(v, *args, **kwargs):
	"""see slidingMinimaIdxs"""
	return slidingMinimaIdxs(-v, *args, **kwargs)


if __name__ == '__main__':
	import doctest
	doctest.testmod()
