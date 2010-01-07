.. -*- mode: rst; fill-column: 78 -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. _chap_tutorial:
.. index:: tutorial

***********************************
A Tutorial Introduction into PyMVPA
***********************************

COME UP WITH SOME INTRO

Import NumPy_ since it is really useful, and also import a number of little
helpers we are going to use in the tutorial.

>>> import numpy as N
>>> from tutorial_lib import *

As a first step we will load an fMRI dataset that is the first subject of the
classic studie of :ref:`Haxby et al. (2001)`. For the sake of simplicity we are
using a helper function that loads and pre-processes the data in a similar way
as it was done in the original study. Later on we will get back to this point
at look at what was done in more detail, but for now it is as simple as:

>>> ds = get_haxby2001_data()

What we get as `ds` is a PyMVPA dataset that contains the fMRI data, and a lot
of additional information which we will investigate later on. In the original
study the authors split the dataset in half (in odd and even runs), and
computed a *pattern of activation* for each stimulus category in each half.
Hence the dataset consists of 16 patterns which are called :term:`sample` in
PyMVPA (one for each of the eight categories in each half of the experiment).
The number of samples in a dataset is equivalent to its length, and can be
queried by:

>>> print len(ds)
16

Most datasets in PyMVPA represented as a two-dimensional array, where the first
axis is the samples axis, and the second axis represents the features of the
dataset. In the Haxby studie the authors used a region of interest (ROI) in the
ventral temporal cortex. For subject 1 this ROI comprised of 577 voxels. Since
the analysis was done on the voxel activation patterns, those voxels are the
actual features of this dataset, and hence we have 577 of them.

>>> print ds.nfeatures
577

We can also access the information via the `shape` property of the dataset:

>>> print ds.shape
(16, 577)

>>> clf = get_haxby2001_clf()
>>> clf.train(ds)
>>> predictions = clf.predict(ds.samples)
>>> N.mean(predictions == ds.sa.labels)
1.0

>>> terr = TransferError(clf)
>>> terr(ds[ds.sa.runtype == 'odd'], ds[ds.sa.runtype == 'even'])
0.125
>>> terr(ds[ds.sa.runtype == 'even'], ds[ds.sa.runtype == 'odd'])
0.0

>>> cvte = CrossValidatedTransferError(terr, splitter=HalfSplitter(attr='runtype'))
>>> cv_results = cvte(ds)
>>> N.mean(cv_results)
0.0625

>>> print cv_results.sa.cv_fold
['odd->even' 'even->odd']
