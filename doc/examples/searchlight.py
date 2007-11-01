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

from optparse import OptionParser

from mvpa.datasets.niftidataset import NiftiDataset
from mvpa.algorithms.clfcrossval import ClfCrossValidation
from mvpa.clf.knn import kNN
from mvpa.datasets.nfoldsplitter import NFoldSplitter
from mvpa.algorithms.searchlight import Searchlight

from mvpa.misc import verbose
from mvpa.misc.cmdline import \
     optsCommon, optRadius, optKNearestDegree, optCrossfoldDegree

usage = """\
%s [options] <NIfTI samples> <labels+blocks> <NIfTI mask>

where labels+blocks is a text file that lists the class label and the
associated block of each data sample/volume as a tuple of two integer
values (separated by a single space). -- one tuple per line.""" \
% sys.argv[0]


parser = OptionParser(usage=usage,
                      option_list=optsCommon + \
                      [optRadius, optKNearestDegree, optCrossfoldDegree])


(options, files) = parser.parse_args()

if len(files)!=3:
    parser.error("Please provide 3 files in the command line")
    sys.exit(1)

verbose(1, "Loading data")

# data filename
dfile = files[0]
# text file with labels and block definitions (chunks)
cfile = files[1]
# mask volume filename
mfile = files[2]

# read conditions into an array (assumed to be two columns of integers)
# TODO: We need some generic helper to read conditions stored in some
#       common formats
verbose(2, "Reading conditions from file %s" % cfile)
cfile = open(cfile, 'r')
conds = N.fromfile(cfile, sep=' ', dtype=int).reshape(2, -1)
cfile.close()

verbose(2, "Loading volume file %s" % dfile)
data = NiftiDataset(dfile, conds[0], conds[1], mfile)

verbose(1, "Computing")

verbose(3, "Assigning a measure to be CrossValidation")
# compute N-1 cross-validation with a kNN classifier in each sphere
cv = ClfCrossValidation(kNN(k=options.knearestdegree),
                        NFoldSplitter(cvtype=options.crossfolddegree))

verbose(3, "Generating Searchlight instance")
# contruct searchlight with 5mm radius
# this assumes that the spatial pixdim values in the source NIfTI file
# are specified in mm
sl = Searchlight(cv, radius=options.radius)

# run searchlight
verbose(3, "Running searchlight on loaded data")
results = sl(data)

print results

# XXX add function to NiftiDataset that calls the mapper and creates a
# new NiftiImage from the results.
