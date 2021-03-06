#!/bin/env/python

import os


# ------------------------------- Filesystem
def ls():
    return os.listdir('.')


def isHidden(path):
    filename = os.path.basename(path)
    return filename.startswith('.')


def isVisible(path):
    return not isHidden(path)


def joinPaths(dir, contents):
    return [os.path.join(dir, f) for f in contents]


def filesInDirMatching(dir, prefix=None, suffix=None, absPaths=False,
                       onlyFiles=False, onlyDirs=False):
    files = os.listdir(dir)
    if prefix:
        files = [f for f in files if f.startswith(prefix)]
    if suffix:
        files = [f for f in files if f.endswith(suffix)]
    if onlyFiles or onlyDirs:
        paths = joinPaths(dir, files)
        if onlyFiles:
            newFiles = []
            for f, path in zip(files, paths):
                if os.path.isfile(path):
                    newFiles.append(f)
            files = newFiles
        if onlyDirs:
            newFiles = []
            for f, path in zip(files, paths):
                if os.path.isdir(path):
                    newFiles.append(f)
            files = newFiles
    if absPaths:
        files = joinPaths(dir, files)
    return files


def listSubdirs(dir, startswith=None, endswith=None, absPaths=False):
    return filesInDirMatching(dir, startswith, endswith, absPaths,
                              onlyDirs=True)


def listFilesInDir(dir, startswith=None, endswith=None, absPaths=False):
    return filesInDirMatching(dir, startswith, endswith, absPaths,
                              onlyFiles=True)


def listHiddenFilesInDir(dir, startswith=None, endswith=None, absPaths=False):
    contents = filesInDirMatching(dir, startswith, endswith, absPaths,
                                  onlyFiles=True)
    return list(filter(isHidden, contents))


def listVisibleFilesInDir(dir, startswith=None, endswith=None, absPaths=False):
    contents = filesInDirMatching(dir, startswith, endswith, absPaths,
                                  onlyFiles=True)
    return list(filter(isVisible, contents))


def ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def basename(f, noexts=False):
    name = os.path.basename(f)
    if noexts:
        name = name.split('.')[0]
    return name
