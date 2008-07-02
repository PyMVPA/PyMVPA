#!/usr/bin/env python
#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Example showing some possibilities of data exploration
(i.e. to 'smell' data).
"""

import numpy as N
import pylab as P

from mvpa.misc.plot import plotFeatureHist, plotSamplesDistance
from mvpa import cfg
from mvpa.datasets.nifti import NiftiDataset
from mvpa.misc.iohelpers import SampleAttributes

# load example fmri dataset
attr = SampleAttributes('data/attributes.txt')
ds = NiftiDataset(samples='data/bold.nii.gz',
                  labels=attr.labels,
                  chunks=attr.chunks,
                  mask='data/mask.nii.gz')

# only use the first 5 chunks to save some cpu-cycles
ds = ds.selectSamples(ds.chunks < 5)

# take a look at the distribution of the feature values in all
# sample categories and chunks
plotFeatureHist(ds, perchunk=True, bins=20, normed=True,
                xlim=(0, ds.samples.max()))
if cfg.getboolean('examples', 'interactive', True):
    P.show()

# next only works with floating point data
ds.setSamplesDType('float')

# look at sample similiarity
# Note, the decreasing similarity with increasing temporal distance
# of the samples
plotSamplesDistance(ds, sortbyattr='chunks')
P.title('Sample distances (sorted by chunks)')
if cfg.getboolean('examples', 'interactive', True):
    P.show()

# similar distance plot, but now samples sorted by their respective labels,
# i.e. samples with same labels are plotted in adjacent columns/rows.
# Note, that the first and largest group corresponds to the 'rest' condition
# in the dataset
plotSamplesDistance(ds, sortbyattr='labels')
P.title('Sample distances (sorted by labels)')
if cfg.getboolean('examples', 'interactive', True):
    P.show()


# XXX add some more, maybe show effect of preprocessing
