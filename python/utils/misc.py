#!/bin/python

import numpy as np
import datetime
import inspect
import hashlib
import uuid

# ================================================================
# Constants
# ================================================================

# ================================================================
# Functions
# ================================================================

# ------------------------------- Debug output

def printVar(name, val):
	print(name + "=")
	print(val)


def nowAsString():
	return datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")


def hashArray(A):
	b = A.view(np.uint8)
	return hashlib.sha1(b).hexdigest()


# ------------------------------- Inspection

def getNumArgs(func):
	(args, varargs, varkw, defaults) = inspect.getargspec(func)
	return len(args)


# ------------------------------- Numerical Funcs

def rnd(A, dec=6):
	return np.round(A, decimals=dec)


# ------------------------------- Randomness

def randStrId(length=4):
	return str(uuid.uuid4())[:length]


# ------------------------------- Testing

def main():
	pass

if __name__ == '__main__':
	main()
