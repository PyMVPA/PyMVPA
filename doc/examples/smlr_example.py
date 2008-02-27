#!/usr/bin/env python
#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Example demonstrating a SMLR classifier"""

import numpy as N
from mvpa.datasets.dataset import Dataset
from mvpa.clfs.smlr import SMLR


nfeat = 10000
nsamp = 100
ntrain = 90
goodfeat = 10
offset = 0.5

samp1 = N.random.randn(nsamp,nfeat)
samp1[:,:goodfeat] += offset
lab1 = N.ones((nsamp,1))
#samp1 = N.hstack((lab1,samp1))

samp2 = N.random.randn(nsamp,nfeat)
samp2[:,:goodfeat] -= offset
lab2 = N.ones((nsamp,1))*2
#samp2 = N.hstack((lab2,samp2))

traindat = N.vstack((samp1[:ntrain,:],samp2[:ntrain,:]))
testdat = N.vstack((samp1[ntrain:,:],samp2[ntrain:,:]))

# create the pymvpa dataset from the labeled features
patternsPos = Dataset(samples=samp1[:ntrain,:], labels=1)
patternsNeg = Dataset(samples=samp2[:ntrain,:], labels=0)
patterns = patternsPos + patternsNeg

# set up the classifier
clf = SMLR()

# enable saving of the values used for the prediction
clf.states.enable('values')

# train with the known points
clf.train(patterns)

# run the predictions on the test values
pre = clf.predict(testdat)
