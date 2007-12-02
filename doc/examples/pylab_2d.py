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
from mvpa.clf.plf import PLF

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
logReg.train(patterns)

# set up the testing features
x1=N.linspace(-10,10, 100)
x2=N.linspace(-10,10, 100)
x,y=N.meshgrid(x1,x2);
feat_test=N.array((N.ravel(x), N.ravel(y)))

# this is need b/c I can't access the __f function in PLF
def logistic(y):
    return 1./(1+N.exp(-y))

# do the regression prediction (this would be ideally part of the
# classifier)
# Michael: maybe something like this would do it (given that Classifier
# starts to inherit 'State' as well.
#
# logReg.predict(feat_test) # ignore the returned predictions
# res = logReg['decision_value'] 
#
# Yarik will come up with a better name ;-) but basically every classifier
# should be able to provide the bit of information that is used to finally
# choose a class label. Actually, for a SVM regression the class label is
# identical to the prediction so in some cases this might be a duplicate.
res = N.ravel(logistic(logReg.offset+feat_test.T*logReg.w))

# reshape the results
z = N.asarray(res).reshape((100,100))

# plot the predictions
pylab.pcolor(x, y, z, shading='interp')
pylab.clim(0,1)
pylab.colorbar()
pylab.contour(x, y, z, linewidths=1, colors='black', hold=True)
pylab.show()
