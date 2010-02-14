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
kNN -- Model Flexibility in Pictures
====================================

.. index:: kNN

TODO

"""

import numpy as N


"""

"""

import mvpa
from mvpa.base import cfg
from mvpa.misc.data_generators import *
from mvpa.clfs.knn import kNN
from mvpa.misc.plot import *

mvpa.seed(0)                            # to reproduce the plot

dataset_kwargs = dict(nfeatures=2, nchunks=10,
    snr=2, nlabels=4, means=[ [0,1], [1,0], [1,1], [0,0] ])

dataset_train = normal_feature_dataset(**dataset_kwargs)
dataset_plot = normal_feature_dataset(**dataset_kwargs)


# make a new figure
P.figure(figsize=(9, 9))

for i,k in enumerate((1, 3, 9, 20)):
    knn = kNN(k)

    print "Processing %s problem..." % id
    P.subplot(2, 2, i+1)

    """
    """

    knn.train(dataset_train)

    plot_decision_boundary_2d(
        dataset_plot, clf=knn, maps='targets')

if cfg.getboolean('examples', 'interactive', True):
    # show all the cool figures
    P.show()
