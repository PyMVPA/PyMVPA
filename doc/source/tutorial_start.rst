.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. vi: set ft=rst sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. index:: Tutorial, Cross-validation
.. _chap_tutorial_start:

**********************
Part 1: A Gentle Start
**********************

.. note::

  This tutorial part is also available for download as an `IPython notebook
  <http://ipython.org/ipython-doc/dev/interactive/htmlnotebook.html>`_:
  [`ipynb <notebooks/tutorial_start.ipynb>`_]

The purpose of this first tutorial part is to make you familiar with a few basic
properties and building blocks of PyMVPA. Let's have a slow start and compute a
:term:`cross-validation` analysis.

Virtually every Python script starts with some ``import`` statements that load
functionality provided elsewhere. We start this tutorial by importing some
little helpers (including all of PyMVPA) we are going to use in the tutorial,
and whose purpose we are going to see shortly.

>>> from mvpa2.tutorial_suite import *

Getting the data
================

As a first step, we will load an fMRI dataset that is the first subject of the
classic study of :ref:`Haxby et al. (2001) <HGF+01>`. For the sake of
simplicity we are using a helper function that loads and pre-processes the data
in a way that is similar to what the original authors did. Later on we will get
back to this point and look at what was done in greater detail, but for now it is
as simple as:

>>> ds = get_haxby2001_data()

What we get as ``ds`` is a PyMVPA dataset that contains the fMRI data, and a lot
of additional information which we will investigate later on. In the original
study the authors split the dataset in half (in odd and even runs), and
computed a *pattern of activation* for each stimulus category in each half.
Hence, the dataset consists of 16 patterns which are called :term:`sample`\s in
PyMVPA (one for each of the eight categories in each half of the experiment).
The number of samples in a dataset is equivalent to its length, and can be
queried by:

>>> print len(ds)
16

Most datasets in PyMVPA are represented as a two-dimensional array, where the first
axis is the samples axis, and the second axis represents the :term:`feature`\s
of the samples. In the Haxby study the authors used a region of interest (ROI)
in the ventral temporal cortex. For subject 1 this ROI comprises 577 voxels.
Since the analysis was done on the voxel activation patterns, those voxels are
the actual features of this dataset, and hence we have 577 of them.

>>> print ds.nfeatures
577

We can also access this information via the
`~mvpa2.base.dataset.AttrDataset.shape` property of the dataset:

>>> print ds.shape
(16, 577)

The most important pieces of information for a classification analysis, besides
the data, are the so-called :term:`label`\s or :term:`target`\s that are
assigned to the samples, since they define the model that should be learned
by a :term:`classifier`, and serve as target values to assess the prediction
accuracy. The dataset stores these targets in its collection of **s**\ample
**a**\ttributes (hence the collection name ``sa``), and they can be accessed by
their attribute name, either through the collection, or via a shortcut.

>>> print ds.sa.targets
['bottle' 'cat' 'chair' 'face' 'house' 'scissors' 'scrambledpix' 'shoe'
 'bottle' 'cat' 'chair' 'face' 'house' 'scissors' 'scrambledpix' 'shoe']
>>> print ds.targets
['bottle' 'cat' 'chair' 'face' 'house' 'scissors' 'scrambledpix' 'shoe'
 'bottle' 'cat' 'chair' 'face' 'house' 'scissors' 'scrambledpix' 'shoe']

As it can be seen, PyMVPA can handle literal labels, and there is no need to
recode them into numerical values.

.. exercise::

  Besides the collection of sample attributes ``sa``, each dataset has two
  additional collections: ``fa`` for feature attributes and ``a`` for general
  dataset attributes. All these collections are actually instances of a
  Python `dict`. Investigate what additional attributes are stored in this
  particular dataset.


Dealing With A Classifier
=========================

All that we are missing for a first attempt at a classification analysis of
this dataset is a :term:`classifier`. This time we will not use a magic
function to help us, but will create the classifier ourselves. The original study
employed a so-called 1-nearest-neighbor classifier, using correlation as a
distance measure. In PyMVPA this type of classifier is provided by the
`~mvpa2.clfs.knn.kNN` class, that makes it possible to specify the desired
parameters.

>>> clf = kNN(k=1, dfx=one_minus_correlation, voting='majority')

A k-Nearest-Neighbor classifier performs classification based on the similarity
of a sample with respect to each sample in a :term:`training dataset`.  The
value of ``k`` specifies the number of neighbors to derive a
prediction, ``dfx`` sets the distance measure that determines the neighbors, and
``voting`` selects a strategy to choose a single label from the set of targets
assigned to these neighbors.

.. exercise::

  Access the built-in help to inspect the ``kNN`` class regarding additional
  configuration options.

Now that we have a classifier instance, it can be easily trained by passing the
dataset to its ``train()`` method.

>>> clf.train(ds)

A trained classifier can subsequently be used to perform classification of
unlabeled samples. The classification can be assessed by comparing these
predictions to the target labels.

>>> predictions = clf.predict(ds.samples)
>>> np.mean(predictions == ds.sa.targets)
1.0

We see that the classifier performs remarkably well on our dataset -- it
doesn't make even a single prediction error. However, most of the time we would
not be particularly interested in the prediction accuracy of a classifier on the
same dataset that it got trained with.

.. exercise::

  Think about why this particular classifier will always perform error-free
  classification of the training data -- regardless of the actual dataset
  content. If the reason is not immediately obvious, take a look at chapter
  13.3 in :ref:`The Elements of Statistical Learning <HTF09>`. Investigate how
  the accuracy varies with different values of ``k``. Why is that?

Instead, we are interested in the generalizability of the classifier on new,
unseen data. This would allow us, in principle, to use it to assign labels to
unlabeled data. Because we only have a single dataset, it needs to be split
into (at least) two parts to achieve this. In the original study, Haxby and
colleagues split the dataset into patterns of activations from odd versus
even-numbered runs. Our dataset has this information in the ``runtype`` sample
attribute:

>>> print ds.sa.runtype
['even' 'even' 'even' 'even' 'even' 'even' 'even' 'even' 'odd' 'odd' 'odd'
 'odd' 'odd' 'odd' 'odd' 'odd']

Using this attribute we can now easily split the dataset in half. PyMVPA
datasets can be sliced in similar ways as NumPy_'s `ndarray`. The following
calls select the subset of samples (i.e. rows in the datasets) where the value
of the ``runtype`` attribute is either the string 'even' or 'odd'.

>>> ds_split1 = ds[ds.sa.runtype == 'odd']
>>> len(ds_split1)
8
>>> ds_split2 = ds[ds.sa.runtype == 'even']
>>> len(ds_split2)
8

Now we could repeat the steps above: call ``train()`` with one dataset half
and ``predict()`` with the other, and compute the prediction accuracy
manually.  However, a more convenient way is to let the classifier do this for
us.  Many objects in PyMVPA support a post-processing step that we can use to
compute something from the actual results. The example below computes the
*mismatch error* between the classifier predictions and the *target* values
stored in our dataset. To make this work, we do not call the classifier's
``predict()`` method anymore, but "call" the classifier directly with the test
dataset. This is a very common usage pattern in PyMVPA that we shall see a lot
over the course of this tutorial.  Again, please note that we compute an error
now, hence lower values represent more accurate classification.

>>> clf.set_postproc(BinaryFxNode(mean_mismatch_error, 'targets'))
>>> clf.train(ds_split2)
>>> err = clf(ds_split1)
>>> print np.asscalar(err)
0.125

In this case, our choice of which half of the dataset is used for training and
which half for testing was completely arbitrary, hence we could also estimate
the transfer error after swapping the roles:

>>> clf.train(ds_split1)
>>> err = clf(ds_split2)
>>> print np.asscalar(err)
0.0

We see that on average the classifier error is really low, and we achieve an
accuracy level comparable to the results reported in the original study.

.. _sec_tutorial_crossvalidation:

Cross-validation
================

What we have just done was manually split the dataset into
combinations of training and testing datasets, given a specific sample attribute
-- in this case whether a *pattern of activation* or
:term:`sample` came from *even* or *odd* runs.  We ran the classification
analysis on each split to estimate the performance of the
classifier model. In general, this approach is called :term:`cross-validation`,
and involves splitting the dataset into multiple pairs of subsets, choosing
sample groups by some criterion, and estimating the classifier performance by
training it on the first dataset in a split and testing against the second
dataset from the same split.

PyMVPA provides a way to allow complete cross-validation procedures to run
fully automatically, without the need for manual splitting of a dataset. Using
the `~mvpa2.measures.base.CrossValidation` class, a cross-validation is set up
by specifying what measure should be computed on each dataset split and how
dataset splits should be generated. The measure that is usually computed is
the transfer error that we already looked at in the previous section. The
second element, a :term:`generator` for datasets, is another very common tool
in PyMVPA. The following example uses
`~mvpa2.generators.partition.HalfPartitioner`, a generator that, when called
with a dataset, marks all samples regarding their association with the first
or second half of the dataset. This happens based on the values of a specified
sample attribute -- in this case ``runtype`` -- much like the manual dataset
splitting that we have performed earlier.
`~mvpa2.generators.partition.HalfPartitioner` will make sure to subsequently
assign samples to both halves, i.e. samples from the first half in the first
generated dataset will be in the second half of the second generated dataset.
With these two techniques we can replicate our manual cross-validation easily
-- reusing our existing classifier, but without the custom post-processing
step.

>>> # disable post-processing again
>>> clf.set_postproc(None)
>>> # dataset generator
>>> hpart = HalfPartitioner(attr='runtype')
>>> # complete cross-validation facility
>>> cv = CrossValidation(clf, hpart)

.. exercise::

  Try calling the ``hpart`` object with our dataset. What happens? Now try
  passing the dataset to its ``generate()`` methods. What happens now?
  Make yourself familiar with the concept of a Python generator. Investigate
  what the code snippet ``list(xrange(5))`` does, and try to adapt it to the
  ``HalfPartitioner``.

Once the ``cv`` object is created, it can be called with a dataset, just like
we did with the classifier before. It will internally perform all the dataset
partitioning, split each generated dataset into training and testing sets
(based on the partitions), and train and test the classifier repeatedly.
Finally, it will return the results of all cross-validation folds.

>>> cv_results = cv(ds)
>>> np.mean(cv_results)
0.0625

Actually, the cross-validation results are returned as another dataset that has
one sample per fold and a single feature with the computed transfer-error per
fold.

>>> len(cv_results)
2
>>> cv_results.samples
array([[ 0.   ],
       [ 0.125]])

..
  Disable for now as this doesn't work that way anymore. Look at RepeatedMeasure
  for a related XXX...
  The advantage of having a dataset as the return value (as opposed to a plain
  vector, or even a single number) is that we can easily attach additional
  information. In this case the dataset also contains some information about
  which samples (indicated by the respective attribute values used by the
  splitter) formed the training and testing datasets in each fold.
  .
  >>> print cv_results.sa.cvfolds
  [0 1]

This could be the end of a very simple introduction to cross-validation with
PyMVPA. However, since we were cheating a bit in the beginning, we actually
still don't know how to import data other than the single subject from the
Haxby study. This is the topic of the :ref:`next chapter <chap_tutorial_datasets>`.

.. _NumPy: http://numpy.scipy.org

.. todo::

  * TEST THE DIFFERENCE OF HALFSPLITTER vs. ODDEVEN SPLITTER on the full dataset later on


References
==========

:ref:`Haxby et al. (2001) <HGF+01>`
  *Classic MVPA study. Its subject 1 serves as the example dataset in this
  tutorial part.*

:ref:`Hastie et al. (2009) <HTF09>`
  *Comprehensive reference of statistical learning methods.*
