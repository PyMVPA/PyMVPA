.. -*- mode: rst; fill-column: 78 -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. _chap_tutorial1:
.. index:: Basic dataset handling

********************************************************
Tutorial Part 1: Datasets, Classifiers, Cross-Validation
********************************************************

COME UP WITH SOME INTRO

Import NumPy_ since it is really useful, and also import a number of little
helpers we are going to use in the tutorial.

>>> import numpy as N
>>> from tutorial_lib import *

As a first step we will load an fMRI dataset that is the first subject of the
classic studie of :ref:`Haxby et al. (2001) <HGF+01>`. For the sake of
simplicity we are using a helper function that loads and pre-processes the data
in a similar way as it was done in the original study. Later on we will get
back to this point at look at what was done in more detail, but for now it is
as simple as:

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
axis is the samples axis, and the second axis represents the :term:`feature`\s of the
dataset. In the Haxby studie the authors used a region of interest (ROI) in the
ventral temporal cortex. For subject 1 this ROI comprised of 577 voxels. Since
the analysis was done on the voxel activation patterns, those voxels are the
actual features of this dataset, and hence we have 577 of them.

>>> print ds.nfeatures
577

We can also access the information via the
`~mvpa.base.dataset.AttrDataset.shape` property of the dataset:

>>> print ds.shape
(16, 577)

The most important information for a classification analysis, besides the data,
are the so-called :term:`label`\s assigned to the samples, since they define
the model that should be learned by a classifier, and serve as target values to
assess the prediction accuracy. The datasets stores these labels in its
collection of sample attributes, and they can be accessed by the attribute
name, either through the collection, or via a shortcut.

>>> print ds.sa.labels
['bottle' 'cat' 'chair' 'face' 'house' 'scissors' 'scrambledpix' 'shoe'
 'bottle' 'cat' 'chair' 'face' 'house' 'scissors' 'scrambledpix' 'shoe']
>>> print ds.labels
['bottle' 'cat' 'chair' 'face' 'house' 'scissors' 'scrambledpix' 'shoe'
 'bottle' 'cat' 'chair' 'face' 'house' 'scissors' 'scrambledpix' 'shoe']

As it can be seen, PyMVPA can handle literal labels, and there is no need to
recode them into numerical values.

All that we are missing for a first attempt of a classification analysis of
this dataset is a :term:`classifier`. This time we will not use a magic
function to help us, but create the classifier ourselves. The original study
employed a so-called 1-nearest-neighbor classifier, using correlation as a
distance measure. In PyMVPA this type of classifier is provided by the
`~mvpa.clfs.knn.kNN` class, that makes is possible to specify the desired
parameters.

>>> clf = kNN(k=1, dfx=oneMinusCorrelation, voting='majority')

A k-Nearest-Neighbor classifier performs classification based on the similarity
of a sample with respect to each sample in a :term:`training dataset`.  The
value of `k` specifies the number of neighbors to shall be used to derive a
prediction, `dfx` sets the distance measure that determines the neighbors, and
`voting` selects a strategy to choose a single label from the set of labels
assigned to these neighbors as the prediction.

Now that we have a classifier instance it can easily be trained by passing the dataset to its `train()` method.

>>> clf.train(ds)

A trained classifier can subsequently be used to perform classifications of
unlabled samples. The classification can be assessed by comparing these
predictions to the target labels.

>>> predictions = clf.predict(ds.samples)
>>> N.mean(predictions == ds.sa.labels)
1.0

We see that the classifier performs remarkably well on our dataset -- it
doesn't make even a single prediction error. However, most of the time we would
not be interested in the prediction accuracy of the classifier on this
particular data, since it is the same dataset that I got trained with.

.. note::

  Think about why this particular classifier will always perform error-free
  classification of the training data -- regardless of the actual dataset
  content.

Instead, we are interested in the generalizabilty of the classifier model on
new, unseen, and most importantly unlabeled data. Since we only have a single
dataset it needs to be split into (at least) two parts to achieve this. In the
original study Haxby and colleagues split the dataset into pattern of
activations from odd versus even-numbered run. Our dataset has this information
in the `runtype` sample attribute:

>>> print ds.sa.runtype
['even' 'even' 'even' 'even' 'even' 'even' 'even' 'even' 'odd' 'odd' 'odd'
 'odd' 'odd' 'odd' 'odd' 'odd']

Using this attribute we can now easily split the dataset into two. PyMVPA
dataset can be sliced in similar ways as NumPy_ or Matlab arrays. The following
calls select the subset of samples (i.e. rows in the datasets), where the value
of the `runtype` attribute is either the string 'even' or 'odd'.

>>> ds_split1 = ds[ds.sa.runtype == 'odd']
>>> len(ds_split1)
8
>>> ds_split2 = ds[ds.sa.runtype == 'even']
>>> len(ds_split2)
8

To conveniently assess the generalization performance of a trained classifier
model on new data, PyMVPA provides the `~mvpa.clfs.transerror.TransferError`
class. It actually doesn't measure the accuracy, but by default the
classification **error** -- and more precisely the fraction of
misclassifications. A `~mvpa.clfs.transerror.TransferError` object is created
by simply providing a classifier that shall be trained on one dataset and
tested against another. In this case, we are going to reuse our kNN classifier
instance. Once created, the generalization error can be computed by calling the
`terr` object with two datasets: The first argument is the :term:`test dataset`
and the second argument is the :term:`training dataset`. When training and
testing are done, the fraction of misclassifications is return. Again, please
note that these are now error, hence lower values represent more accurate
classification.

>>> terr = TransferError(clf)
>>> terr(ds_split1, ds_split2)
0.125

In this case, our choice of training dataset and test dataset was completely
arbitrary, hence we also estimate the transfer error after swapping the roles:

>>> terr(ds_split2, ds_split1)
0.0

We see that on average the classifier error is really low, and we achieve an
accuracy level comparable to the results reported in the original study.

CONTINUE WITH CVTE

>>> cvte = CrossValidatedTransferError(terr, splitter=HalfSplitter(attr='runtype'))
>>> cv_results = cvte(ds)
>>> N.mean(cv_results)
0.0625

>>> print cv_results.sa.cv_fold
['odd->even' 'even->odd']


References
==========

Literature
----------
* :ref:`Haxby et al (2001) <HGF+01>`


Related API Documentation
-------------------------
.. autosummary::
   :toctree:

   ~mvpa.datasets.base.Dataset
   ~mvpa.clfs.knn.kNN


.. _NumPy: http://numpy.scipy.org
