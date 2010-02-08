.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. index:: Tutorial
.. _chap_tutorial4:

**********************************************
Part 4: Classifier -- All Alike, Yet Different
**********************************************

This is already the second time that we will engage in a classification
analysis, so let's first recap what we did before in the :ref:`first tutorial
part <chap_tutorial1>`:

>>> from tutorial_lib import *
>>> ds = get_haxby2001_data()
>>> clf = kNN(k=1, dfx=oneMinusCorrelation, voting='majority')
>>> terr = TransferError(clf)
>>> cvte = CrossValidatedTransferError(terr, splitter=HalfSplitter(attr='runtype'))
>>> cv_results = cvte(ds)
>>> N.mean(cv_results)
0.0625

Looking at this little code snippet we can nicely see the logical parts of
a cross-validated classification analysis.

1. Load the data.
2. Choose a classifier.
3. Set up an error function.
4. Evaluate the error in a cross-validation procedure
5. Inspect results.

Our previous choice of the classifier was guided by the intention to
replicated the :ref:`Haxby et al. (2001) <HGF+01>`, but what if we want to
try a different algorithm? In this case a nice feature of PyMVPA comes into
play. All classifiers implement a common interface that makes them easily
exchangeable without the need to adapt any other part of the analysis code.
If, for example, we want to try the popular :term:`support vector machine`
(SVM) on our example dataset it looks like this:

>>> clf = LinearCSVMC()
>>> terr = TransferError(clf)
>>> cvte = CrossValidatedTransferError(terr, splitter=HalfSplitter(attr='runtype'))
>>> cv_results = cvte(ds)
>>> N.mean(cv_results)
0.1875

Instead of k-nearest-neighbor, we create a linear SVM classifier using the
internally using the popular LIBSVM library (note the PyMVPA provides
additional SVM implementations). The rest of the code remains identical.
SVM with its default settings seems to perform slightly worse than the
simple kNN-classifier. We'll get back to the classifiers shortly. Let's
first look at the remaining part of this analysis.

We already know that `~mvpa.clfs.transerror.TransferError` is used to
compute the error function. So far we have only used the mean mismatch
between actual targets and classifier predictions as error function.
However, PyMVPA offers a number of alternative functions in the
:mod:`mvpa.misc.errorfx` module, but it is only trivial to specify custom
ones. For example, if we do not want to have error reported, but accuracy,
we can do that:

>>> terr = TransferError(clf, errorfx=lambda p, t: N.mean(p == t))
>>> cvte = CrossValidatedTransferError(terr, splitter=HalfSplitter(attr='runtype'))
>>> cv_results = cvte(ds)
>>> N.mean(cv_results)
0.8125

This example reused the SVM classifier we have create before, and
yields exactly what we expect from the previous result.

The details of the cross-validation procedure are also heavily
customizable. We have seen that a `~mvpa.datasets.splitters.Splitter` is
used to generate training and testing dataset for each cross-validation
fold. So far we have only used `~mvpa.datasets.splitters.HalfSplitter` to
divide the dataset into odd and even runs (based on our custom sample
attribute ``runtype``. However, in general it is more common to perform so
called leave-one-out cross-validation, where *one* independent part of a
dataset is selected as testing dataset, while all other remain as the
training dataset. This procedure is repeated till all parts have served as
the testing dataset once. In case of our dataset we could consider each of
the 12 runs as independent measurements (fMRI data doesn't allow us to
consider temporally adjacent data to be considered independent). To set up
a 12-fold leave-one-run-out cross-validation, we can make use of
`~mvpa.datasets.splitters.NFoldSplitter`:

>>> cvte = CrossValidatedTransferError(terr, splitter=NFoldSplitter())
>>> cv_results = cvte(ds)
#>>> print cv_results.samples

DATASET HAS ONLY ODD AND EVEN LEFT

* prediction
* estimate
* ca, parameters
* meta classifiers...

.. exercise::

   Try doing the Z-Scoring beforce computing the mean samples per category.
   What happens to the generalization performance of the classifier?
   ANSWER: It becomes 100%!


.. only:: html

  References
  ==========

  .. autosummary::
     :toctree: generated

     ~mvpa.clfs.base.Classifier
