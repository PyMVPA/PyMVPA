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
import pylab as P

# local imports
from mvpa.datasets.dataset import Dataset
from mvpa.clfs.plf import PLF
from mvpa.clfs.ridge import RidgeReg
from mvpa.clfs.svm import RbfNuSVMC,LinearNuSVMC

# set up the labeled data
# two skewed 2-D distributions
num_dat = 200
dist = 4
feat_pos=N.random.randn(2, num_dat)
feat_pos[0, :] *= 2.
feat_pos[1, :] *= .5
feat_pos[0, :] += dist
feat_neg=N.random.randn(2, num_dat)
feat_neg[0, :] *= .5
feat_neg[1, :] *= 2.
feat_neg[0, :] -= dist

# set up the testing features
x1 = N.linspace(-10, 10, 100)
x2 = N.linspace(-10, 10, 100)
x,y = N.meshgrid(x1, x2);
feat_test = N.array((N.ravel(x), N.ravel(y)))

# create the pymvpa dataset from the labeled features
patternsPos = Dataset(samples=feat_pos.T, labels=1)
patternsNeg = Dataset(samples=feat_neg.T, labels=0)
patterns = patternsPos + patternsNeg

# set up classifiers to try out
clfs = {'Ridge Regression': RidgeReg(),
        'Linear SVM': LinearNuSVMC(probability=1),
        'RBF SVM': RbfNuSVMC(probability=1),
        'Logistic Regression': PLF(criterion=0.00001)}

# loop over classifiers and show how they do
fig = 0

# make a new figure
P.figure()
for c in clfs:
    # tell which one we are doing
    print "Running %s classifier..." % (c)

    # make a new subplot for each classifier
    fig += 1
    P.subplot(2,2,fig)

    # plot the training points
    P.plot(feat_pos[0, :], feat_pos[1, :], "r.")
    P.plot(feat_neg[0, :], feat_neg[1, :], "b.")

    # select the clasifier
    clf = clfs[c]

    # enable saving of the values used for the prediction
    clf.states.enable('values')

    # train with the known points
    clf.train(patterns)

    # run the predictions on the test values
    pre = clf.predict(feat_test.T)

    # if ridge, use the prediction, otherwise use the values
    if c == 'Ridge Regression':
        # use the prediction
        res = N.asarray(pre)
    elif c == 'Logistic Regression':
        # get out the values used for the prediction
        res = N.asarray(clf.values)
    else:
        # get the probabilities from the svm
        res = N.asarray([(q[1][1] - q[1][0] + 1) / 2 for q in clf.values])

    # reshape the results
    z = N.asarray(res).reshape((100, 100))

    # plot the predictions
    P.pcolor(x, y, z, shading='interp')
    P.clim(0, 1)
    P.colorbar()
    P.contour(x, y, z, linewidths=1, colors='black', hold=True)

    # add the title
    P.title(c)

# show all the cool figures
P.show()
