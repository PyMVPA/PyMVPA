.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:
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

The purpose of this first tutorial part is to make your familiar with a few basic
properties and building blocks of PyMVPA. Let's have a slow start and compute a
cross-validation analysis.

Virtually every Python script starts with some ``import`` statements that load
functionality provided elsewhere. We start this tutorial by importing some
little helpers (including all of PyMVPA) we are going to use in the tutorial,
and whose purpose we are going to see shortly.

>>> from tutorial_lib import *

Getting the data
================

As a first step we will load an fMRI dataset that is the first subject of the
classic study of :ref:`Haxby et al. (2001) <HGF+01>`. For the sake of
simplicity we are using a helper function that loads and pre-processes the data
in a way similar to the original study. Later on we will get
back to this point and look in greater detail at what was done, but for now it is
as simple as:

>>> ds = get_haxby2001_data()

What we get as ``ds`` is a PyMVPA dataset that contains the fMRI data, and a lot
of additional information which we will investigate later on. In the original
study the authors split the dataset in half (in odd and even runs), and
computed a *pattern of activation* for each stimulus category in each half.
Hence the dataset consists of 16 patterns which are called :term:`sample`\s in
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

We can also access the information via the
`~mvpa.base.dataset.AttrDataset.shape` property of the dataset:

>>> print ds.shape
(16, 577)

The most important information for a classification analysis, besides the data,
are the so-called :term:`label`\s or :term:`target`\s that are assigned to the
samples, since they define the model that should be learned by a
:term:`classifier`, and serve as target values to assess the prediction
accuracy. The dataset stores these targets in its collection of **s**\ample
**a**\ttributes (hence collection name ``sa``), and they can be accessed by the
attribute name, either through the collection, or via a shortcut.

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
  dataset attributes. All these collections are actually instances of a Python
  `dict`. Investigate what additional attributes are stored in this particular
  dataset.


Dealing With A Classifier
=========================

All that we are missing for a first attempt of a classification analysis of
this dataset is a :term:`classifier`. This time we will not use a magic
function to help us, but will create the classifier ourselves. The original study
employed a so-called 1-nearest-neighbor classifier, using correlation as a
distance measure. In PyMVPA this type of classifier is provided by the
`~mvpa.clfs.knn.kNN` class, that makes it possible to specify the desired
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

Now that we have a classifier instance it can easily be trained by passing the
dataset to its ``train()`` method.

>>> clf.train(ds)

A trained classifier can subsequently be used to perform classifications of
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
unseen data so we could, in principle, use it to label unlabeled data. Since
we only have a single dataset it needs to be split into (at least) two parts
to achieve this. In the original study Haxby and colleagues split the dataset
into pattern of activations from odd versus even-numbered runs. Our dataset
has this information in the ``runtype`` sample attribute:

>>> print ds.sa.runtype
['even' 'even' 'even' 'even' 'even' 'even' 'even' 'even' 'odd' 'odd' 'odd'
 'odd' 'odd' 'odd' 'odd' 'odd']

Using this attribute we can now easily split the dataset into two. PyMVPA
datasets can be sliced in similar ways as NumPy_'s `ndarray`. The following
calls select the subset of samples (i.e. rows in the datasets), where the value
of the ``runtype`` attribute is either the string 'even' or 'odd'.

>>> ds_split1 = ds[ds.sa.runtype == 'odd']
>>> len(ds_split1)
8
>>> ds_split2 = ds[ds.sa.runtype == 'even']
>>> len(ds_split2)
8

To conveniently assess the generalization performance of a trained classifier
model on new data, PyMVPA provides the `~mvpa.clfs.transerror.TransferError`
class. It actually doesn't measure the accuracy, but by default the
classification **error** (more precisely the fraction of misclassifications). A
`~mvpa.clfs.transerror.TransferError` instance is created by simply providing a
classifier that shall be trained on one dataset and tested against another. In
this case, we are going to reuse our kNN classifier instance. Once created, the
generalization error can be computed by calling the ``terr`` object with two
datasets: The first argument is the :term:`testing dataset` and the second
argument is the :term:`training dataset`. When training and testing is done,
the fraction of misclassifications is returned. Again, please note that this is
now an error, hence lower values represent more accurate classification.

>>> terr = TransferError(clf)
>>> terr(ds_split1, ds_split2)
0.125

In this case, our choice of which half of the dataset is used for training and
which half for testing was completely arbitrary, hence we also estimate the
transfer error after swapping the roles:

>>> terr(ds_split2, ds_split1)
0.0

We see that on average the classifier error is really low, and we achieve an
accuracy level comparable to the results reported in the original study.

Cross-validation
================

What we have just done manually, was splitting the dataset into
combinations of training and testing datasets, given a specific sample attribute
-- in this case the information whether a *pattern of activation* or
:term:`sample` came from *even* or *odd* runs.  We ran the classification
analysis on each split to estimate the performance of the
classifier model. In general, this approach is called :term:`cross-validation`,
and involves splitting the dataset in multiple pairs of subsets, choosing
sample groups by some criterion, and estimating the classifier performance by
training it on the first dataset in a split and testing against the second
dataset from the same split.

PyMVPA provides a class to allow complete cross-validation procedures to run
automatically, without the need for manual splitting of a dataset. Using the
`~mvpa.algorithms.cvtranserror.CrossValidatedTransferError` class a
cross-validation is set up by specifying what measure should be computed on
each dataset split, and how dataset splits shall be generated. The measure that
is usually computed is the transfer error that we already looked at in the
previous section. For dataset splitting PyMVPA provides various
`~mvpa.datasets.splitters.Splitter` classes. To replicate our manual
cross-validation, we can simply reuse the ``terr`` instance as our measure, and
use a so-called `~mvpa.datasets.splitters.HalfSplitter` to generate the desired
dataset splits. Note, that the splitter is instructed to use the ``runtype``
attribute to determine which samples should form a dataset subset.

>>> hspl = HalfSplitter(attr='runtype')
>>> cvte = CrossValidatedTransferError(terr, splitter=hspl)

.. exercise::

  Try calling the ``hspl`` object with our dataset. What happens? How can we
  get the split datasets from it?

Once the ``cvte`` object is created, it can be called with a dataset and
will internally perform all splitting, as well as training and testing on each
split generated by the splitter. Finally it will return the results of all
cross-validation folds.

>>> cv_results = cvte(ds)
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

The advantage of having a dataset as the return value (as opposed to a plain
vector, or even a single number) is that we can easily attach additional
information. In this case the dataset also contains some information about
which samples (indicated by the respective attribute values used by the
splitter) formed the training and testing datasets in each fold.

>>> print cv_results.sa.cv_fold
['odd->even' 'even->odd']

This could be the end of a very simple introduction into cross-validation with
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


.. only:: html

   .. autosummary::
      :toctree: generated

      ~mvpa.algorithms.cvtranserror.CrossValidatedTransferError
      ~mvpa.datasets.base.Dataset
      ~mvpa.clfs.knn.kNN
      mvpa.datasets.splitters
      ~mvpa.clfs.transerror.TransferError


