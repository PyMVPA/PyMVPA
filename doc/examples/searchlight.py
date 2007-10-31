#!/usr/bin/python
#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: Example demonstrating a searchlight analysis on an fMRI dataset"""

import sys

import numpy as N
import scipy.signal

from mvpa.datasets.niftidataset import NiftiDataset
from mvpa.algorithms.clfcrossval import ClfCrossValidation
from mvpa.clf.knn import kNN
from mvpa.datasets.nfoldsplitter import NFoldSplitter
from mvpa.algorithms.searchlight import Searchlight

if not len(sys.argv) == 4:
    print """\
Usage: %s <NIfTI samples> <labels+blocks> <NIfTI mask>

where labels+blocks is a text file that lists the class label and the
associated block of each data sample/volume as a tuple of two integer
values (separated by a single space). -- one tuple per line.""" \
        % sys.argv[0]
    sys.exit(1)

# data filename
dfile = sys.argv[1]
# text file with labels and block definitions (chunks)
cfile = sys.argv[2]
# mask volume filename
mfile = sys.argv[3]

# read conditions into an array (assumed to be two columns of integers)
cfile = open(cfile, 'r')
conds = N.fromfile(cfile, sep=' ', dtype=int).reshape(2,-1)
cfile.close()

data = NiftiDataset(dfile, conds[0], conds[1], mfile)

# compute N-1 cross-validation with a kNN classifier in each sphere
cv = ClfCrossValidation(kNN(k=3),
                        NFoldSplitter(cvtype=1))

# contruct searchlight with 5mm radius
# this assumes that the spatial pixdim values in the source NIfTI file
# are specified in mm
sl = Searchlight(cv, radius=5.0, verbose=True)

# run searchlight
results = sl(data)

print results

# XXX add function to NiftiDataset that calls the mapper and creates a new NiftiImage
# from the results.
