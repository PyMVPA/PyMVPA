.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. vi: set ft=rst sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. index:: Tutorial, Classifier, meta-classifier
.. _chap_tutorial_meta_classifiers:

**********************************************
 Classifiers that do more -- Meta Classifiers
**********************************************

.. note::

  This tutorial part is also available for download as an `IPython notebook
  <http://ipython.org/ipython-doc/dev/interactive/htmlnotebook.html>`_:
  [`ipynb <notebooks/tutorial_meta_classifiers.ipynb>`_]

In :ref:`chap_tutorial_classifiers` we saw that it is possible to encapsulate a
whole cross-validation analysis into a single object that can be called with
any dataset to produce the desired results. We also saw that despite this
encapsulation we can still get a fair amount of information about the performed
analysis.  However, what happens if we want to do some further processing of
the data **within** the cross-validation analysis. That seems to be difficult,
since we feed a whole dataset into the analysis, and only internally does it
get split into the respective pieces.

Of course there is a solution to this problem -- a :term:`meta-classifier`.
This is a classifier that doesn't implement a classification algorithm on
its own, but uses another classifier to do the actual work. In addition,
the meta-classifier adds another processing step that is performed before
the actual :term:`base-classifier` sees the data.

An example of such a meta-classifier is `~mvpa2.clfs.meta.MappedClassifier`.
Its purpose is simple: Apply a mapper to both training and testing data
before it is passed on to the internal base-classifier. With this technique
it is possible to implement arbitrary pre-processing within a
cross-validation analysis.

Before we get into that, let's reproduce the dataset from
:ref:`chap_tutorial_classifiers`:

>>> from mvpa2.tutorial_suite import *
>>> # directory that contains the data files
>>> datapath = os.path.join(tutorial_data_path, 'data')
>>> # load the raw data
>>> ds = load_tutorial_data(roi='vt')
>>> # pre-process
>>> poly_detrend(ds, polyord=1, chunks_attr='chunks')
>>> zscore(ds, param_est=('targets', ['rest']))
>>> ds = ds[ds.sa.targets != 'rest']
>>> # average
>>> run_averager = mean_group_sample(['targets', 'chunks'])
>>> ds = ds.get_mapped(run_averager)
>>> ds.shape
(96, 577)

Now, suppose we want to perform the classification not on voxel intensities
themselves, but on the same samples in the space spanned by the singular
vectors of the training data, it would look like this:

>>> baseclf = LinearCSVMC()
>>> metaclf = MappedClassifier(baseclf, SVDMapper())
>>> cvte = CrossValidation(metaclf, NFoldPartitioner())
>>> cv_results = cvte(ds)
>>> print np.mean(cv_results)
0.15625

First we notice that little has been changed in the code and the results --
the error is slightly reduced, but still comparable. The critical line is
the second, where we create the `~mvpa2.clfs.meta.MappedClassifier` from the
SVM classifier instance, and a `~mvpa2.mappers.svd.SVDMapper` that
implements `singular value decomposition`_ as a mapper.

.. exercise::

   What might be the reasons for the error decrease in comparison to the
   results on the dataset with voxel intensities?

.. _singular value decomposition: http://en.wikipedia.org/wiki/Singular_value_decomposition

We know that mappers can be combined into complex processing pipelines, and
since `~mvpa2.clfs.meta.MappedClassifier` takes any mapper as argument, we
can implement arbitrary preprocessing steps within the cross-validation
procedure. Let's say we have heard rumors that only the first two dimensions
of the space spanned by the SVD vectors cover the "interesting" variance
and the rest is noise. We can easily check that with an appropriate mapper:

>>> mapper = ChainMapper([SVDMapper(), StaticFeatureSelection(slice(None, 2))])
>>> metaclf = MappedClassifier(baseclf, mapper)
>>> cvte = CrossValidation(metaclf, NFoldPartitioner())
>>> cv_results = cvte(ds)
>>> svm_err = np.mean(cv_results)
>>> print round(svm_err, 2)
0.57

Well, obviously the discarded components cannot only be noise, since the error
is substantially increased. But maybe it is the classifier that cannot deal with
the data. Since nothing in this code is specific to the actual classification
algorithm we can easily go back to the kNN classifier that has served us well
in the past.

>>> baseclf = kNN(k=1, dfx=one_minus_correlation, voting='majority')
>>> mapper = ChainMapper([SVDMapper(), StaticFeatureSelection(slice(None, 2))])
>>> metaclf = MappedClassifier(baseclf, mapper)
>>> cvte = CrossValidation(metaclf, NFoldPartitioner())
>>> cv_results = cvte(ds)
>>> np.mean(cv_results) < svm_err
False

Oh, that was even worse. We would have to take a closer look at the data to
figure out what is happening here.

.. exercise::

   Inspect the confusion matrix of this analysis for both classifiers. What
   information is represented in the first two SVD components and what is not?
   Plot the samples of the full dataset after they have been mapped onto the
   first two SVD components. Why does the kNN classifier perform so bad in
   comparison to the SVM (hint: think about the distance function)?

In this tutorial part we took a look at classifiers. We have seen that,
regardless of the actual algorithm, all classifiers are implementing the same
interface. Because of this, they can be replaced by another classifier without
having to change any other part of the analysis code. Moreover, we have seen
that it is possible to enable and access optional information that is offered
by particular parts of the processing pipeline.

.. Think about adding a demo of the classifiers warehouse.
  .. exercise::
     Try doing the Z-Scoring before computing the mean samples per category.
     What happens to the generalization performance of the classifier?
     ANSWER: It becomes 100%!
