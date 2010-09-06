.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. vi: set ft=rst sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


.. _chap_faq:

**************************
Frequently Asked Questions
**************************

General
=======

.. index:: Matlab


I'm a Matlab user. How hard is learning Python and PyMVPA for me?
-----------------------------------------------------------------

If you are coming from Matlab, you will soon notice a lot of similarities
between Matlab and Python (besides the huge advantages of Python over
Matlab).  For an easy transition you might want to have a look at a `basic
comparison of Matlab and NumPy`_.

.. _basic comparison of Matlab and NumPy: http://www.scipy.org/NumPy_for_Matlab_Users

.. index:: MVPA toolbox for Matlab

It would be nice to have some guidelines on how to use PyMVPA for users who
are already familiar with the `Matlab MVPA toolbox`_. If you are using both
packages and could compile a few tips, your contribution would be most
welcome.

.. _Matlab MVPA toolbox: http://www.csbmb.princeton.edu/mvpa/

A recent paper by :ref:`Jurica and van Leeuwen (2009) <JL09>` describes an
open-source MATLAB®-to-Python compiler which might be a very useful tool to
migrate a substantial amount of Matlab-based source code to Python and
therefore also aids the migration of developers from Matlab to the new
*"general open-source lingua franca for scientific computation"*.


.. index:: optimization

It is sloooooow. What can I do?
-------------------------------

Have you tried running the Python interpreter with `-O`? PyMVPA provides
lots of debug messages with information that is computed in addition to the
work that really has to be done. However, if Python is running in
*optimized* mode, PyMVPA will not waste time on this and really tries to be
fast.

If you are already running it optimized, then maybe you are doing something
really demanding...


I am tired of writing these endless import blocks. Any alternative?
-------------------------------------------------------------------

Sure. Instead of individually importing all pieces that are required
by a script, you can import them all at once. A simple:

  >>> import mvpa.suite as mvpa

makes everything directly accessible through the mvpa namespace, e.g.
`mvpa.datasets.base.Dataset` becomes `mvpa.Dataset`. Really lazy people
can even do:

  >>> from mvpa.suite import *

However, as always there is a price to pay for this convenience. In contrast
to the individual imports there is some initial performance and memory cost. In
the worst case you'll get all external dependencies loaded (e.g. a full R
session), just because you have them installed. Therefore, it might be better
to limit this use to case where individual key presses matter and use
individual imports for production scripts.


I feel like I want to contribute something, do you mind?
--------------------------------------------------------

Not at all! If you think there is something that is not well explained in
the documentation, send us an improvement. If you implemented a new algorithm
using PyMVPA that you want to share, please share. If you have an idea for
some other improvement (e.g. speed, functionality), but you have no
time/cannot/do not want to implement it yourself, please post your idea to
the PyMVPA mailing list.


.. index:: Git, development

I want to develop a new feature for PyMVPA. How can I do it efficiently?
------------------------------------------------------------------------

The best way is to use Git for both, getting the latest code from the
repository and preparing the patch. Here is a quick sketch of the workflow.

First get the latest code::

  git clone git://github.com/PyMVPA/PyMVPA.git

This will create a new `PyMVPA` subdirectory, that contains the complete
repository. Enter this directory and run `gitk --all` to browse the full
history and *all* branches that have ever been published.

You can run::

  git fetch origin

in this directory at any time to get the latest changes from the main
repository.

Next, you have to decide what you want to base your new feature on. In the
simplest case this is the `master` branch (the one that contains the code that
will become the next release). Creating a local branch based on the (remote)
`master` branch is::

  git checkout -b my_hack origin/master

Now you are ready to start hacking. You are free to use all powers of Git
(and yours, of course). You can do multiple commits, fetch new stuff from the
repository, and merge it into your local branch, ... To get a feeling what can
be done, take a look `very short description of Git`_ or `a more
comprehensive Git tutorial`_.

.. _very short description of Git: http://sysmonblog.co.uk/misc/git_by_example/
.. _a more comprehensive Git tutorial: http://www-cs-students.stanford.edu/~blynn/gitmagic/

When you are done with the new feature, you can prepare the patch for
inclusion into PyMVPA. If you have done multiple commits you might want to
squash them into a single patch containing the new feature. You can do this
with `git rebase`.  Any recent version of `git rebase` has an option
`--interactive`, which allows you to easily pick, squash or even further edit
any of the previous commits you have made. Rebase your local branch against
the remote branch you started hacking on (`origin/master` in this example)::

  git rebase --interactive origin/master

When you are done, you can generate the final patch file::

  git format-patch origin/master

Above command will generate a file for each commit in you local branch that is
not yet part of `origin/master`. The patch files can then be easily emailed.



The manual is quite insufficient. When will you improve it?
-----------------------------------------------------------

Writing a manual can be a tricky task if you already know the details and
have to imagine what might be the most interesting information for someone
who is just starting. If you feel that something is missing which has cost
you some time to figure out, please drop us a note and we will add it as
soon as possible. If you have developed some code snippets to demonstrate
some feature or non-trivial behavior (maybe even trivial ones, which are
not as obvious as they should be), please consider sharing this snippet with
us and we will put it into the example collection or the manual. Thanks!


Data import, export and storage
===============================

What file formats are understood by PyMVPA?
-------------------------------------------

Please see the :ref:`data_formats` section.


What if there is no special file format for some particular datatype?
---------------------------------------------------------------------

With the :class:`~mvpa.misc.io.hamster.Hamster` class, PyMVPA
supports storing *any* kind of serializable data into a
(compressed) file (see the class documentation for a trivial
usage example). The facility is particularly useful for storing
any number of intermediate analysis results, e.g. for
post-processing.


Data preprocessing
==================

.. index:: invariant features

Is there an easy way to remove invariant features from a dataset?
-----------------------------------------------------------------

You might have to deal with invariant features in case like an fMRI dataset,
where the *brain mask* is slightly larger than the thresholded fMRI
timeseries image. Such invariant features (i.e. features with zero variance)
are sometime a problem, e.g. they will lead to numerical difficulties when
z-scoring the features of a dataset (i.e. division by zero).

The `mvpa.datasets.miscfx` module provides a convenience function
`remove_invariant_features()` that strips such features from a dataset.


.. index:: Block-averaging

How can I do :term:`block-averaging` of my block-design fMRI dataset?
---------------------------------------------------------------------

The easiest way is to use a mapper to transform/average the respective
samples. Suppose you have a dataset:

  >>> dataset = normal_feature_dataset()
  >>> print dataset
  <Dataset: 100x4@float64, <sa: chunks,targets>>

Averaging all samples with the same label in each chunk individually is done
by applying a mapper to the dataset.

  >>> from mvpa.mappers.fx import mean_group_sample
  >>>
  >>> m = mean_group_sample(['targets', 'chunks'])
  >>> mapped_dataset = dataset.get_mapped(m)
  >>> print mapped_dataset
  <Dataset: 10x4@float64, <sa: chunks,targets>, <a: mapper>>

`mean_group_sample` creates an `FxMapper` that applies a function to
every group of samples in each chunk individually and therefore yields
one sample of each label per chunk.



Data analysis
=============

.. index:: feature selection, feature_ids

How do I know which features were finally selected by a classifier doing feature selection?
-------------------------------------------------------------------------------------------

All classifier possess a conditional attribute `feature_ids`. When enable, the
classifier stores the ids of all features that were finally used to train
the classifier.

  >>> clf = FeatureSelectionClassifier(
  ...           kNN(k=5),
  ...           SensitivityBasedFeatureSelection(
  ...               SMLRWeights(SMLR(lm=1.0), postproc=maxofabs_sample()),
  ...               FixedNElementTailSelector(1, tail='upper', mode='select')),
  ...           enable_ca = ['feature_ids'])
  >>> clf.train(dataset)
  >>> final_dataset = dataset[:, clf.ca.feature_ids]
  >>> print final_dataset
  <Dataset: 100x1@float64, <sa: chunks,targets>>

In the above code snippet a kNN classifier is defined, that performs a feature
selection step prior training. Features are selected according to the maximum
absolute magnitude of the weights of a SMLR classifier trained on the data
(same training data that will also go into kNN). Absolute SMLR weights are used
for feature selection as large negative values also indicate important
information. Finally, the classifier is configured to select the single most
important feature (given the SMLR weights). After enabling the `feature_ids`
state, the classifier provides the desired information, that can e.g. be
applied to generate a stripped dataset for an analysis of the similarity
structure.


.. index:: sensitivity, cross-validation

How do I extract sensitivities from a classifier used within a cross-validation?
--------------------------------------------------------------------------------

.. The answer depends on size of the classification problem and the used
   classifier. If you can afford to keep a copy of the trained classifier for
   each data split, the most elegant solution is probably a :class:`~mvpa.clfs.meta.SplitClassifier`...
   ...BUT no yet

:class:`~mvpa.algorithms.cvtranserror.CrossValidatedTransferError` provides an
interface to access any classifier-related information: `harvest_attribs`.
Harvesting the sensitivities computed by all classifiers (without recomputing
them again) looks like this:

  >>> cv = CrossValidatedTransferError(
  ...       TransferError(SMLR()),
  ...       OddEvenSplitter(),
  ...       harvest_attribs=\
  ...        ['transerror.clf.get_sensitivity_analyzer(force_training=False)()'])
  >>> merror = cv(dataset)
  >>> sensitivities = cv.ca.harvested.values()[0]
  >>> len(sensitivities)
  2
  >>> sensitivities[0].shape == (len(dataset.uniquetargets), dataset.nfeatures)
  True

First, we define an instance of
:class:`~mvpa.algorithms.cvtranserror.CrossValidatedTransferError` that uses an
`~mvpa.clfs.smlr.SMLR` classifier to perform the cross-validation on odd-even
splits of a dataset.  The important piece is the definition of the
`harvest_attribs`.  It takes a list of code snippets that will be executed in
the local context of the cross-validation function. The
:class:`~mvpa.clfs.transerror.TransferError` instance used to train and test
the classifier on each split is available via `transerror`. The rest is easy:
:class:`~mvpa.clfs.transerror.TransferError` provides access to its classifier
and any classifier can in turn generate an appropriate
:class:`~mvpa.measures.base.Sensitivity` instance via
`get_sensitivity_analyzer()`.  This generator method takes additional arguments
to the constructor of the :class:`mvpa.measures.base.Sensitivity` class. In
this case we want to prevent retraining the classifiers, as they will be
trained anyway by the :class:`~mvpa.clfs.transerror.TransferError` instance
they belong to.

The return values of all code snippets defined in `harvest_attribs` are
available in the `harvested` conditional attribute. `harvested` is a dictionary where
the keys are the code snippets used to compute the value. As the key in this
case is pretty long, we simply take the first (and only) value from the
dictionary.  The value is actually a list of sensitivity datasets, one per
split. In each dataset we have, in this case, a per class sensitivity vector.


.. _faq_literal_labels:

Can PyMVPA deal with literal class labels?
------------------------------------------

Yes. For all external machine learning libraries that do not support literal
labels, PyMVPA will transparently convert them to numerical ones, and also
revert this transformation for all output values.
