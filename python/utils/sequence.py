#!usr/bin/env/python

import itertools
import re
from collections import Counter
import numpy as np
from inspect import getargspec
import types


def isDict(x):
	return isinstance(x, dict)

def isListOrTuple(x):
	return isinstance(x, (list, tuple))

def asListOrTuple(x):
	return x if isListOrTuple(x) else [x]

def isString(x):
	return isinstance(x, types.StringTypes)

def flattenListOfLists(l):
	return list(itertools.chain.from_iterable(l))

def makeImmutable(x):
	"""
	>>> makeImmutable(5) == 5
	True
	>>> makeImmutable('a') == 'a'
	True
	>>> makeImmutable((1, 2)) == (1, 2)
	True
	>>> makeImmutable([1, 2]) == [1, 2]
	False
	"""
	# must either be not a collections or immutable
	try:
		{}[x] = 0	# dicts require immutability
		return x
	except TypeError:
		# so it's mutable; either a collection or a
		# mutable class; if a class, we're hosed, so
		# assume it's a collection
		try:
			# if it's a singleton collection, try returning
			# first element; this will jump to except
			# unless x is a collection
			if len(x) == 1:
				return makeImmutable(x[0])

			# not a singleton collection, but still a collection,
			# so make it a tuple
			return tuple(x)
		except TypeError:
			return x 	# not a collection

def isImmutable(x):
	return x == makeImmutable(x)

def asKey(x):
	return makeImmutable(x)

def getNumArgs(func):
	(args, varargs, varkw, defaults) = getargspec(func)
	return len(args)

def elementsAtIdxs(seq, idxs):
	elements = []
	for idx in idxs:
		elements.append(seq[idx])
	return elements

def applyToDict(func, dictionary):
	for key, val in dictionary.iteritems():
		dictionary[key] = func(key, val)
	return dictionary

def where(conditionFunc, iterable):
	"""
	>>> a = [1,3,2,4]
	>>> where(lambda el: el > 1, a)
	[1, 2, 3]
	>>> where(lambda i, el: el > 1 and i < 3, a)
	[1, 2]
	"""
	idxs = []
	nargs = getNumArgs(conditionFunc)
	if nargs == 1:
		for i, el in enumerate(iterable):
			if (conditionFunc(el)):
				idxs.append(i)
	elif nargs == 2:
		for i, el in enumerate(iterable):
			if (conditionFunc(i, el)):
				idxs.append(i)
	return idxs

def whereSubseq(conditionFunc, seq, length, overlap=False):
	"""
	>>> a = [1, 3, 2, 3, 2]
	>>> b = [1, 2]
	>>> whereSubseq(lambda seq: seq == b, a, 2)
	[]
	>>> c = [3, 2]
	>>> whereSubseq(lambda seq: seq == c, a, 2)
	[1, 3]
	>>> whereSubseq(lambda seq: sum(seq) < 8, a, 3)
	[0]
	>>> whereSubseq(lambda seq: sum(seq) < 8, a, 3, overlap=True)
	[0, 2]
	"""
	if (conditionFunc is None) or (seq is None):
		return []
	if (length < 1) or (len(seq) < length):
		return []

	end = len(seq) - length + 1
	allIdxs = []
	if overlap:
		for i in range(end):
			subSeq = seq[i:(i + length)]
			if conditionFunc(subSeq):
				allIdxs.append(i)
	else:
		i = 0
		while i < end:
			subSeq = seq[i:(i + length)]
			if conditionFunc(subSeq):
				allIdxs.append(i)
				i += length 	# skip possible overlaps
			else:
				i += 1

	return allIdxs

def rangesBetweenWhere(conditionFunc, seq, windowLength, overlap=True):
	"""
	>>> a = [1, 1, 4, 3]
	>>> rangesBetweenWhere(lambda s: s[0] > s[1], a, 2)
	... # doctest: +NORMALIZE_WHITESPACE
	array([[0, 2], [2, 4]])
	>>> rangesBetweenWhere(lambda s: s[0] > 10, a, 1)
	array([0, 4])
	"""
	# so basically we just return the starts of all subseqs for which some
	# predicate is true, prepending 0 if it isn't there anyway; basically,
	# if we return idxs and the predicate is p, [idxs[i], idxs[i+1]) is a
	# range in which p() is only true at idxs[i]

	if seq is None or len(seq) == 0:
		return np.array([])
	if windowLength >= len(seq):
		return np.hstack([0, len(seq)])
	idxs = whereSubseq(conditionFunc, seq, windowLength, overlap)
	idxs = np.array(idxs)

	# if it never happened, no splits, so return the whole seq
	if (len(idxs) == 0):
		return np.hstack([0, len(seq)])

	# conceptually, 0 is always the start of a new range
	if (idxs[0] != 0):
		idxs = np.hstack([0, idxs])

	startIdxs = idxs
	endIdxs = np.hstack([idxs[1:], len(seq)])
	return np.vstack([startIdxs, endIdxs])

def rangesOfConstantValue(seq):
	"""
	>>> a = [1, 1, 4, 3]
	>>> rangesOfConstantValue(a)
	... # doctest: +NORMALIZE_WHITESPACE
	array([[0, 2], [2, 3], [3, 4]])
	>>> b = [1, 2, 2, 2]
	>>> rangesOfConstantValue(b)
	... # doctest: +NORMALIZE_WHITESPACE
	array([[0, 1], [1, 4]])
	>>> c = [1]
	>>> rangesOfConstantValue(c)
	... # doctest: +NORMALIZE_WHITESPACE
	array([[0, 1]])
	"""
	if seq is None or len(seq) == 0:
		return np.array([])

	starts = [0]
	ends = []
	for i in xrange(1, len(seq)):
		if seq[i] != seq[i-1]:
			starts.append(i)
			ends.append(i)
	ends.append(len(seq))

	return np.array(zip(starts, ends))


def splitIdxsBy(conditionFunc, seq):
	"""
	>>> a = [1, 3, 2, 2, 1, 3]
	>>> splitIdxsBy(lambda el: el + 10, a)
	{11: [0, 4], 12: [2, 3], 13: [1, 5]}
	>>> splitIdxsBy(lambda i, el: i > el, a)
	{False: [0, 1, 2], True: [3, 4, 5]}
	"""
	if (conditionFunc is None) or (seq is None):
		return []

	bins = {}
	nargs = getNumArgs(conditionFunc)
	if nargs == 1:
		for i, el in enumerate(seq):
			bin = conditionFunc(el)
			key = makeImmutable(bin)
			vals = bins.get(key, [])
			vals.append(i)
			bins[key] = vals
	elif nargs == 2:
		for i, el in enumerate(seq):
			bin = conditionFunc(i, el)
			key = makeImmutable(bin)
			vals = bins.get(key, [])
			vals.append(i)
			bins[key] = vals

	return bins

def splitElementsBy(conditionFunc, seq):
	"""
	Group by conditionFunc(i, seq[i]) or conditionFunc(seq[i])

	>>> a = [1, 3, 2, 2, 1, 3]
	>>> splitElementsBy(lambda i, el: i > 2, a)
	{False: [1, 3, 2], True: [2, 1, 3]}
	>>> splitElementsBy(lambda el: el % 2, a)
	{0: [2, 2], 1: [1, 3, 1, 3]}
	"""
	bins = splitIdxsBy(conditionFunc, seq)
	# for key in bins.keys():
	# 	bins[key] = elementsAtIdxs(seq, bins[key])
	# return bins
	return applyToDict(lambda k, v: elementsAtIdxs(seq, v), bins)

def splitSubseqsBy(conditionFunc, seq, subSeqLength):
	"""
	Splits seq into nbins lists based on the value returned by
	conditionFunc(seq[i:(i+subSeqLength)]).

	>>> a = [1, 3, 2, 2, 1, 3]
	>>> length = 1
	>>> splitSubseqsBy(lambda s: s + 10, a, length)
	{11: [1, 1], 12: [2, 2], 13: [3, 3]}
	>>> b = [3, 1]
	>>> length = 2
	>>> splitSubseqsBy(lambda s: s[-1] == 3, a, length)
	{False: [(3, 2), (2, 2), (2, 1)], True: [(1, 3), (1, 3)]}
	"""
	subseqs = allSubseqsOfLength(seq, subSeqLength)
	return splitElementsBy(conditionFunc, subseqs)

def findSubseq(subSeq, fullSeq, overlap=False):
	"""
	>>> q = 'aba'
	>>> s = 'cababa'
	>>> findSubseq(q, s)
	[1]
	>>> findSubseq(q, s, overlap=True)
	[1, 3]
	>>> a = [1, 3, 2, 3, 2, 3]
	>>> b = [3, 2, 3]
	>>> findSubseq(b, a)
	[1]
	>>> findSubseq(b, a, overlap=True)
	[1, 3]
	"""
	if (not isString(subSeq)) or (not isString(fullSeq)):
		return whereSubseq(lambda s: s == subSeq, fullSeq,
			len(subSeq), overlap)

	# if we know it's a string, use regexes for speed (but mostly so
	# that I have an excuse to keep this code snippet around, honestly)
	if overlap:
		query = '(?=%s)' % subSeq
		return [m.start() for m in re.finditer(query, fullSeq)]
	else:
		return [m.start() for m in re.finditer(subSeq, fullSeq)]

def calcSurroundingIdxs(idxs, before, after, idxLimit=None):
	"""
	Returns (idxs - before), (idxs + after), where elements of
	idxs that result in values < 0 or > idxLimit are removed; basically,
	this is useful for extracting ranges around certain indices (say,
	matches for a substring) that are contained fully in an array.

	>>> idxs = [1, 3, 4]
	>>> calcSurroundingIdxs(idxs, 0, 0)
	([1, 3, 4], [1, 3, 4])
	>>> calcSurroundingIdxs(idxs, 1, 0)
	([0, 2, 3], [1, 3, 4])
	>>> calcSurroundingIdxs(idxs, 1, 1)
	([0, 2, 3], [2, 4, 5])
	>>> calcSurroundingIdxs(idxs, 1, 1, 4)
	([0, 2], [2, 4])
	"""
	if idxLimit is None:
		idxs = filter(lambda idx: (idx - before >= 0), idxs)
	else:
		idxs = filter(lambda idx: (idx - before >= 0) and (idx + after <= idxLimit),
			idxs)
	beforeIdxs = map(lambda x: x - before, idxs)
	afterIdxs = map(lambda x: x + after, idxs)
	return beforeIdxs, afterIdxs

def findSurroundingIdxs(subSeq, fullSeq, before, after,
	overlap=False):
	"""
	For each index of subSeq within fullSeq, returns (index - before)
	and (index + after) as elements of beforeIdxs and afterIdxs lists,
	respectively. Overlaps are disallowed in determining matches
	of subSeq unless overlap=True.
	"""
	idxs = findSubseq(subSeq, fullSeq, overlap)
	return calcSurroundingIdxs(idxs, before, after, len(fullSeq))

def extractSubseqs(seq, startIdxs, stopIdxs):
	return map(lambda x, y: seq[x:y], startIdxs, stopIdxs)

def extractSurroundingSubseqs(subSeq, fullSeq, before, after,
	overlap=False):
	"""
	For each index of subSeq within fullSeq, returns
	fullSeq[(index - before):(index + after)].
	"""
	beforeIdxs, afterIdxs = findSurroundingIdxs(subSeq, fullSeq,
		before, after, overlap)
	return extractSubseqs(fullSeq, beforeIdxs, afterIdxs)

def extractPredecessorsWithLength(subSeq, fullSeq, length,
	overlap=False):
	# fullSeq[(matchIndex - length) : (matchIndex)]
	return extractSurroundingSubseqs(subSeq, fullSeq, length, 0, overlap)

def extractSuccessorsWithLength(subSeq, fullSeq, length,
	overlap=False):
	# fullSeq[(matchIndex+1) : (matchIndex+length+1)]
	return extractSurroundingSubseqs(subSeq, fullSeq, -1, length+1, overlap)

def extractSubseqsWhere(conditionFunc, seq, subSeqLength,
	overlap=False):
	startIdxs = whereSubseq(conditionFunc, seq, subSeqLength, overlap)
	stopIdxs = map(lambda x: x + subSeqLength, startIdxs)
	return extractSubseqs(seq, startIdxs, stopIdxs)

def numMatches(query, collection):
	"""returns the number of elements of collection that equal query"""
	return len([s for s in collection if s == query])

# def countXfollowsY(x, y, fullSeq, overlap=False):
# 	"""
# 	Returns the number of times subsequence x immediately follows
# 	subsequence y within the sequence fullSeq
# 	"""
# 	yIdxs = findSubseq(y, fullSeq, overlap)
# 	startIdxs, stopIdxs = calcSurroundingIdxs(yIdxs, -1, len(x)+1, len(fullSeq))
# 	sucessors = extractSubseqs(fullSeq, startIdxs, stopIdxs)
# 	matches = numMatches(x, sucessors)
# 	return matches

def allSubseqsOfLength(seq, length):
	"""
	>>> a = [1, 2, 3]
	>>> allSubseqsOfLength(a, 1)
	[1, 2, 3]
	>>> allSubseqsOfLength(a, 2)
	[(1, 2), (2, 3)]
	>>> allSubseqsOfLength(a, 3)
	[(1, 2, 3)]
	"""
	if (seq is None) or (length is None) or (len(seq) < length):
		return []
	if length == 1:
		return seq

	subseqs = []
	numSubseqs = len(seq) - length + 1
	for i in range(numSubseqs):
		subseq = seq[i: i + length]
		subseqs.append(makeImmutable(subseq))
	return subseqs

def uniqueElementPositions(iterable):
	"""
	Returns a mapping of unique elements to positions at which they
	occur within the iterable
	"""
	objs2positions = {}
	for i, obj in enumerate(iterable):
		key = asKey(obj)
		positions = objs2positions.get(key, [])
		positions.append(i)
		objs2positions[key] = positions
	return objs2positions

def uniqueElementCounts(iterable):
	"""
	Returns a Counter (dict subclass) mapping unique elements to
	how many times they occur within the iterable
	"""
	iterable = map(makeImmutable, iterable)
	return Counter(iterable)

def uniqueSubseqsPositions(seq, length):
	"""
	Return a mapping of all unique subsequences of the provided length
	within seq to the positions at which they occur therein
	"""
	return uniqueElementPositions(allSubseqsOfLength(seq, length))

def uniqueSubseqsCounts(seq, length):
	"""
	Return a Counter (dict subclass) of all unique subsequences of the
	provided length within seq to the number of times they occur therein
	"""
	return uniqueElementCounts(allSubseqsOfLength(seq, length))

def uniqueElements(seq):
	return uniqueElementCounts(seq).keys()

def uniqueSubseqsPreceding(subSeq, fullSeq, length, overlap=False):
	"""
	Returns a mapping of unique subsequences s
	within fullSeq that precede subSeq to how many times each s
	occurs within fullSeq

	# 'abc' happens twice, 'dec' happens once
	>>> d = uniqueSubseqsPreceding('c', 'abcdecabc', 2)
	>>> len(d)
	2
	>>> d['ab']
	2
	>>> d['de']
	1
	"""
	predecessors = extractPredecessorsWithLength(subSeq, fullSeq,
		length, overlap)
	return uniqueElementCounts(predecessors)

def predecessorCounts(subSeq, fullSeq, length, overlap=False):
	"""
	Returns a mapping of (unique subsequences of the given length that
	occur immediately preceding subSeq within fullSeq) to (numpy array
	containing [the number of times that subSeq occurs after the subsequence
	in fullSeq, total number of times subSeq occurs in fullSeq].

	E.g.:
	# 'abc' happens twice, 'abd' happens once
	>>> predecessorCounts('c', 'abcabdabc', 2)
	{'ab': array([2, 3])}
	"""
	# predecessors = extractPredecessorsWithLength(subSeq, fullSeq,
	# 	length, overlap)
	precedingCounts = uniqueSubseqsPreceding(subSeq, fullSeq,
		length, overlap)
	allSeqCounts = uniqueSubseqsCounts(fullSeq, length)

	seqs2counts = {}
	for key, countPrecede in precedingCounts.iteritems():
		countTotal = allSeqCounts[key]
		seqs2counts[key] = np.array((countPrecede, countTotal))

	return seqs2counts

if __name__ == '__main__':
	import doctest
	doctest.testmod()

	s = 'abcabdabcefc'
	seqs2counts = predecessorCounts('c', s, 2)
	assert(len(seqs2counts.keys()) == 2)
	assert(np.array_equal( seqs2counts['ab'], np.array([2, 3]) ))
	assert(np.array_equal( seqs2counts['ef'], np.array([1, 1]) ))

	# a = [1, 1, 4, 3]
	# print rangesOfConstantValue(a)
	# array([0, 2], [2, 3], [3, 4])
	# b = [1, 2, 2, 2]
	# print rangesOfConstantValue(b)
	# # array([0, 1], [1, 4])
	# c = [1]
	# print rangesOfConstantValue(c)
	# # array([0, 1])
