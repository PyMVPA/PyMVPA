#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Simple Plotting of Classifier Behavior
======================================

.. index:: plotting example

This example runs a number of classifiers on a simple 2D dataset and plots the
decision surface of each classifier.

First compose some sample data -- no PyMVPA involved.
"""

import numpy as np

# set up the labeled data
# two skewed 2-D distributions
num_dat = 200
dist = 4
# Absolute max value allowed. Just to assure proper plots
xyamax = 10
feat_pos=np.random.randn(2, num_dat)
feat_pos[0, :] *= 2.
feat_pos[1, :] *= .5
feat_pos[0, :] += dist
feat_pos = feat_pos.clip(-xyamax, xyamax)
feat_neg=np.random.randn(2, num_dat)
feat_neg[0, :] *= .5
feat_neg[1, :] *= 2.
feat_neg[0, :] -= dist
feat_neg = feat_neg.clip(-xyamax, xyamax)

# set up the testing features
npoints = 101
x1 = np.linspace(-xyamax, xyamax, npoints)
x2 = np.linspace(-xyamax, xyamax, npoints)
x,y = np.meshgrid(x1, x2);
feat_test = np.array((np.ravel(x), np.ravel(y)))

"""Now load PyMVPA and convert the data into a proper
:class:`~mvpa2.datasets.base.Dataset`."""

from mvpa2.suite import *

# create the pymvpa dataset from the labeled features
patternsPos = dataset_wizard(samples=feat_pos.T, targets=1)
patternsNeg = dataset_wizard(samples=feat_neg.T, targets=0)
ds_lin = vstack((patternsPos, patternsNeg))

"""Let's add another dataset: XOR. This problem is not linear separable
and therefore need a non-linear classifier to be solved. The dataset is
provided by the PyMVPA dataset warehouse.
"""

# 30 samples per condition, SNR 2
ds_nl = pure_multivariate_signal(30, 2)
l1 = ds_nl.sa['targets'].unique[1]

datasets = {'linear': ds_lin, 'non-linear': ds_nl}

"""This demo utilizes a number of classifiers. The instantiation of a
classifier involves almost no runtime costs, so it is easily possible
compile a long list, if necessary."""

# set up classifiers to try out
clfs = {
        'Ridge Regression': RidgeReg(),
        'Linear SVM': LinearNuSVMC(probability=1,
                      enable_ca=['probabilities']),
        'RBF SVM': RbfNuSVMC(probability=1,
                      enable_ca=['probabilities']),
        'SMLR': SMLR(lm=0.01),
        'Logistic Regression': PLR(criterion=0.00001),
        '3-Nearest-Neighbour': kNN(k=3),
        '10-Nearest-Neighbour': kNN(k=10),
        'GNB': GNB(common_variance=True),
        'GNB(common_variance=False)': GNB(common_variance=False),
        'LDA': LDA(),
        'QDA': QDA(),
        }

# How many rows/columns we need
nx = int(ceil(np.sqrt(len(clfs))))
ny = int(ceil(len(clfs)/float(nx)))

"""Now we are ready to run the classifiers. The following loop trains
and queries each classifier to finally generate a nice plot showing
the decision surface of each individual classifier, both for the linear and
the non-linear dataset."""

for id, ds in datasets.iteritems():
    # loop over classifiers and show how they do
    fig = 0

    # make a new figure
    pl.figure(figsize=(nx*4, ny*4))

    print "Processing %s problem..." % id

    for c in sorted(clfs):
        # tell which one we are doing
        print "Running %s classifier..." % (c)

        # make a new subplot for each classifier
        fig += 1
        pl.subplot(ny, nx, fig)

        # select the clasifier
        clf = clfs[c]

        # enable saving of the estimates used for the prediction
        clf.ca.enable('estimates')

        # train with the known points
        clf.train(ds)

        # run the predictions on the test values
        pre = clf.predict(feat_test.T)

        # if ridge, use the prediction, otherwise use the values
        if c == 'Ridge Regression':
            # use the prediction
            res = np.asarray(pre)
        elif 'Nearest-Ne' in c:
            # Use the dictionaries with votes
            res = np.array([e[l1] for e in clf.ca.estimates]) \
                  / np.sum([e.values() for e in clf.ca.estimates], axis=1)
        elif c == 'Logistic Regression':
            # get out the values used for the prediction
            res = np.asarray(clf.ca.estimates)
        elif c in ['SMLR']:
            res = np.asarray(clf.ca.estimates[:, 1])
        elif c in ['LDA', 'QDA'] or c.startswith('GNB'):
            # Since probabilities are logprobs -- just for
            # visualization of trade-off just plot relative
            # "trade-off" which determines decision boundaries if an
            # alternative log-odd value was chosen for a cutoff
            res = np.asarray(clf.ca.estimates[:, 1]
                             - clf.ca.estimates[:, 0])
            # Scale and position around 0.5
            res = 0.5 + res/max(np.abs(res))
        else:
            # get the probabilities from the svm
            res = np.asarray([(q[1][1] - q[1][0] + 1) / 2
                    for q in clf.ca.probabilities])

        # reshape the results
        z = np.asarray(res).reshape((npoints, npoints))

        # plot the predictions
        pl.pcolor(x, y, z, shading='interp')
        pl.clim(0, 1)
        pl.colorbar()
        # plot decision surfaces at few levels to emphasize the
        # topology
        pl.contour(x, y, z, [0.1, 0.4, 0.5, 0.6, 0.9],
                   linestyles=['dotted', 'dashed', 'solid', 'dashed', 'dotted'],
                   linewidths=1, colors='black', hold=True)

        # plot the training points
        pl.plot(ds.samples[ds.targets == 1, 0],
               ds.samples[ds.targets == 1, 1],
               "r.")
        pl.plot(ds.samples[ds.targets == 0, 0],
               ds.samples[ds.targets == 0, 1],
               "b.")

        pl.axis('tight')
        # add the title
        pl.title(c)

if cfg.getboolean('examples', 'interactive', True):
    # show all the cool figures
    pl.show()
