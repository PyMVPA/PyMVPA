#!/usr/bin/env python
#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Example demonstrating a simple classifiction of a 2-D dataset"""

import numpy as N
import pylab

# local imports
from mvpa.datasets.dataset import Dataset
from mvpa.clfs.plf import PLF

# set up the labeled data
# two skewed  2-D distributions
num_dat=200
dist = 4
feat_pos=N.random.randn(2,num_dat)
feat_pos[0,:] *= 2.
feat_pos[1,:] *= .5
feat_pos[0,:] += dist
feat_neg=N.random.randn(2,num_dat)
feat_neg[0,:] *= .5
feat_neg[1,:] *= 2.
feat_neg[0,:] -= dist

# plot those points
pylab.clf()
pylab.plot(feat_pos[0,:], feat_pos[1,:], "r.") ;
pylab.plot(feat_neg[0,:], feat_neg[1,:], "b.") ;

# create the pymvpa dataset from the labeled features
patternsPos = Dataset(samples=feat_pos.T,labels=1)
patternsNeg = Dataset(samples=feat_neg.T,labels=0)
patterns = patternsPos+patternsNeg

# set up the classifier
logReg = PLF(criterion=0.00001)

# enable saving of the values used for the prediction
logReg.enableState('values')

# train with the known points
logReg.train(patterns)

# set up the testing features
x1=N.linspace(-10,10, 100)
x2=N.linspace(-10,10, 100)
x,y=N.meshgrid(x1,x2);
feat_test=N.array((N.ravel(x), N.ravel(y)))

# run the predictions on the test values
pre = logReg.predict(feat_test.T)

# get out the values used for the prediction
res = logReg['values']

# reshape the results
z = N.asarray(res).reshape((100,100))

# plot the predictions
pylab.pcolor(x, y, z, shading='interp')
pylab.clim(0,1)
pylab.colorbar()
pylab.contour(x, y, z, linewidths=1, colors='black', hold=True)
pylab.show()
