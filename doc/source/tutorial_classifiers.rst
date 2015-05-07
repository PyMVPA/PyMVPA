.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. vi: set ft=rst sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. index:: Tutorial, Classifier
.. _chap_tutorial_classifiers:

*****************************************
 Classifiers -- All Alike, Yet Different
*****************************************

.. note::

  This tutorial part is also available for download as an `IPython notebook
  <http://ipython.org/ipython-doc/dev/interactive/htmlnotebook.html>`_:
  [`ipynb <notebooks/tutorial_classifiers.ipynb>`_]

In this chapter we will continue our work from :ref:`chap_tutorial_mappers` in
order to replicate the study of :ref:`Haxby et al. (2001) <HGF+01>`. For this
tutorial there is a little helper function to yield the dataset we generated
manually before:

>>> from mvpa2.tutorial_suite import *
>>> ds = get_haxby2001_data()

The original study employed a so-called 1-nearest-neighbor classifier, using
correlation as a distance measure. In PyMVPA this type of classifier is
provided by the `~mvpa2.clfs.knn.kNN` class, that makes it possible to specify
the desired parameters.

>>> clf = kNN(k=1, dfx=one_minus_correlation, voting='majority')

A k-Nearest-Neighbor classifier performs classification based on the similarity
of a sample with respect to each sample in a :term:`training dataset`.  The
value of ``k`` specifies the number of neighbors to derive a prediction,
``dfx`` sets the distance measure that determines the neighbors, and ``voting``
selects a strategy to choose a single label from the set of targets assigned to
these neighbors.

.. exercise::

  Access the built-in help to inspect the ``kNN`` class regarding additional
  configuration options.

Now that we have a classifier instance, it can be easily trained by passing the
dataset to its ``train()`` method.

>>> clf.train(ds)

A trained classifier can subsequently be used to perform classification of
unlabeled samples. The classification performance can be assessed by comparing
these predictions to the target labels.

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

.. index:: cross-validation
.. _sec_tutorial_crossvalidation:

Cross-validation
================

What we have just done was to manually split the dataset into combinations of
training and testing datasets, given a specific sample attribute -- in this
case whether a *pattern of activation* or :term:`sample` came from *even* or
*odd* runs.  We ran the classification analysis on each split to estimate the
performance of the classifier model. In general, this approach is called
:term:`cross-validation`, and involves splitting the dataset into multiple
pairs of subsets, choosing sample groups by some criterion, and estimating the
classifier performance by training it on the first dataset in a split and
testing against the second dataset from the same split.

PyMVPA provides a way to allow complete cross-validation procedures to run
fully automatic, without the need for manual splitting of a dataset. Using the
`~mvpa2.measures.base.CrossValidation` class, a cross-validation is set up by
specifying what measure should be computed on each dataset split and how
dataset splits should be generated. The measure that is usually computed is the
transfer error that we already looked at in the previous section. The second
element, a :term:`generator` for datasets, is another very common tool in
PyMVPA. The following example uses
`~mvpa2.generators.partition.HalfPartitioner`, a generator that, when called
with a dataset, marks all samples regarding their association with the first or
second half of the dataset. This happens based on the values of a specified
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

.. _NumPy: http://numpy.scipy.org

.. todo::

  * TEST THE DIFFERENCE OF HALFSPLITTER vs. ODDEVEN SPLITTER on the full dataset later on

Any classifier, really
======================

A short summary of all code for the analysis we developed so far is this:

>>> clf = kNN(k=1, dfx=one_minus_correlation, voting='majority')
>>> cvte = CrossValidation(clf, HalfPartitioner(attr='runtype'))
>>> cv_results = cvte(ds)
>>> np.mean(cv_results)
0.0625

Looking at this little code snippet we can nicely see the logical parts of
a cross-validated classification analysis.

1. Load the data
2. Choose a classifier
3. Set up an error function
4. Evaluate the error in a cross-validation procedure
5. Inspect results

Our previous choice of the classifier was guided by the intention to replicate
:ref:`Haxby et al. (2001) <HGF+01>`, but what if we want to try a different
algorithm? In this case another nice feature of PyMVPA comes into play. All
classifiers implement a common interface that makes them easily interchangeable
without the need to adapt any other part of the analysis code.  If, for
example, we want to try the popular :mod:`support vector machine
<mvpa2.clfs.svm>` (SVM) on our example dataset it looks like this:

>>> clf = LinearCSVMC()
>>> cvte = CrossValidation(clf, HalfPartitioner(attr='runtype'))
>>> cv_results = cvte(ds)
>>> np.mean(cv_results)
0.1875

Instead of k-nearest-neighbor, we create a linear SVM classifier,
internally using the popular LIBSVM library (note that PyMVPA provides
additional SVM implementations). The rest of the code remains identical.
SVM with its default settings seems to perform slightly worse than the
simple kNN-classifier. We'll get back to the classifiers shortly. Let's
first look at the remaining part of this analysis.

We already know that `~mvpa2.measures.base.CrossValidation` can be used to
compute errors. So far we have only used the mean number of mismatches between
actual targets and classifier predictions as the error function (which is the
default).  However, PyMVPA offers a number of alternative functions in the
:mod:`mvpa2.misc.errorfx` module, but it is also trivial to specify custom
ones.  For example, if we do not want to have error reported, but instead
accuracy, we can do that:

>>> cvte = CrossValidation(clf, HalfPartitioner(attr='runtype'),
...                        errorfx=lambda p, t: np.mean(p == t))
>>> cv_results = cvte(ds)
>>> np.mean(cv_results)
0.8125

This example reuses the SVM classifier we have create before, and
yields exactly what we expect from the previous result.

The details of the cross-validation procedure are also heavily
customizable. We have seen that a `~mvpa2.generators.partition.Partitioner` is
used to generate training and testing dataset for each cross-validation
fold. So far we have only used `~mvpa2.generators.partition.HalfPartitioner` to
divide the dataset into odd and even runs (based on our custom sample
attribute ``runtype``). However, in general it is more common to perform so
called leave-one-out cross-validation, where *one* independent part of a
dataset is selected as testing dataset, while the other parts constitute the
training dataset. This procedure is repeated till all parts have served as
the testing dataset once. In case of our dataset we could consider each of
the 12 runs as independent measurements (fMRI data doesn't allow us to
consider temporally adjacent data to be considered independent).

To run such an analysis, we first need to redo our dataset preprocessing,
as, in the current one, we only have one sample per stimulus category for
both odd and even runs. To get a dataset with one sample per stimulus
category for each run, we need to modify the averaging step. Using what we
have learned from the :ref:`last tutorial part <chap_tutorial_mappers>` the
following code snippet should be plausible:

>>> # directory that contains the data files
>>> datapath = os.path.join(tutorial_data_path, 'haxby2001')
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

Instead of two samples per category in the whole dataset, now we have one
sample per category, per experiment run, hence 96 samples in the whole
dataset. To set up a 12-fold leave-one-run-out cross-validation, we can
make use of `~mvpa2.generators.partition.NFoldPartitioner`. By default it is
going to select samples from one ``chunk`` at a time:

>>> cvte = CrossValidation(clf, NFoldPartitioner(),
...                        errorfx=lambda p, t: np.mean(p == t))
>>> cv_results = cvte(ds)
>>> np.mean(cv_results)
0.78125

We get almost the same prediction accuracy (reusing the SVM classifier and
our custom error function). Note that this time we performed the analysis on
a lot more samples that were each was computed from just a few fMRI volumes
(about nine each).

So far we have just looked at the mean accuracy or error. Let's investigate
the results of the cross-validation analysis a bit further.

>>> type(cv_results)
<class 'mvpa2.datasets.base.Dataset'>
>>> print cv_results.samples
[[ 0.75 ]
 [ 0.875]
 [ 1.   ]
 [ 0.75 ]
 [ 0.75 ]
 [ 0.875]
 [ 0.75 ]
 [ 0.875]
 [ 0.75 ]
 [ 0.375]
 [ 1.   ]
 [ 0.625]]

The returned value is actually a `~mvpa2.datasets.base.Dataset` with the
results for all cross-validation folds. Since our error function computes
only a single scalar value for each fold the dataset only contains a single
feature (in this case the accuracy), and a sample per each fold.

..
  XXX disabled for now -- see tutorial_start for reason
  Moreover, the dataset also offers a sample attribute that show which particular
  set of chunks formed the training and testing set per fold.
  .
  >> print cv_results.sa.cvfold
  ['1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0->0.0'
   '0.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0->1.0'
   '0.0,1.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0->2.0'
   '0.0,1.0,2.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0->3.0'
   '0.0,1.0,2.0,3.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0->4.0'
   '0.0,1.0,2.0,3.0,4.0,6.0,7.0,8.0,9.0,10.0,11.0->5.0'
   '0.0,1.0,2.0,3.0,4.0,5.0,7.0,8.0,9.0,10.0,11.0->6.0'
   '0.0,1.0,2.0,3.0,4.0,5.0,6.0,8.0,9.0,10.0,11.0->7.0'
   '0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,9.0,10.0,11.0->8.0'
   '0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,10.0,11.0->9.0'
   '0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,11.0->10.0'
   '0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0->11.0']


We Need To Take A Closer Look
=============================

By now we have already done a few cross-validation analyses using two
different classifiers and different pre-processing strategies. In all these
cases we have just looked at the generalization performance or error.
However, error rates hide a lot of interesting information that is very
important for an interpretation of results. In our case we analyzed a
dataset with eight different categories. An average misclassification rate
doesn't tell us much about the contribution of each category to the
prediction error. It could be that *half of the samples of each category*
get misclassified, but the same average error might be due to *all samples
from half of the categories* being completely misclassified, while
prediction accuracy for samples from the remaining categories is perfect.
These two results would have to be interpreted in totally different ways,
despite the same average error rate.

In psychological research this type of results is usually presented as a
`contingency table`_ or `cross tabulation`_ of expected vs. empirical
results. `Signal detection theory`_ offers a whole range of techniques to
characterize such results. From this angle a
classification analysis is hardly any different from a psychological
experiment where a human observer performs a detection task, hence the same
analysis procedures can be applied here as well.

.. _contingency table: http://en.wikipedia.org/wiki/Contingency_table
.. _cross tabulation: http://en.wikipedia.org/wiki/Cross_tabulation
.. _signal detection theory: http://en.wikipedia.org/wiki/Detection_theory

PyMVPA provides convenient access to :term:`confusion matrices <confusion
matrix>`, i.e.  contingency tables of targets vs. actual predictions.  However,
to prevent wasting CPU-time and memory they are not computed by default, but
instead have to be enabled explicitly. Optional analysis results like this are
available in a dedicated collection of :term:`conditional attribute`\ s --
analogous to ``sa`` and ``fa`` in datasets, it is named ``ca``. Let's see how
it works:

>>> cvte = CrossValidation(clf, NFoldPartitioner(),
...                        errorfx=lambda p, t: np.mean(p == t),
...                        enable_ca=['stats'])
>>> cv_results = cvte(ds)

Via the ``enable_ca`` argument we triggered computing confusion tables for
all cross-validation folds, but otherwise there is no change in the code.
Afterwards the aggregated confusion for the whole cross-validation
procedure is available in the ``ca`` collection. Let's take a look (note
that in the printed manual the output is truncated due to page-width
constraints -- please refer to the HTML-based version full the full matrix).

>>> print cvte.ca.stats.as_string(description=True)
----------.
predictions\targets     bottle         cat          chair          face         house        scissors    scrambledpix      shoe
            `------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ P'   N'   FP   FN   PPV  NPV  TPR  SPC  FDR  MCC  F1
       bottle             6             0             3             0             0             5             0             1       15   75    9    6   0.4 0.92  0.5 0.88  0.6 0.34 0.44
        cat               0             10            0             0             0             0             0             0       10   67    0    2    1  0.97 0.83   1    0  0.79 0.91
       chair              0             0             7             0             0             0             0             0        7   73    0    5    1  0.93 0.58   1    0  0.66 0.74
        face              0             2             0             12            0             0             0             0       14   63    2    0  0.86   1    1  0.97 0.14  0.8 0.92
       house              0             0             0             0             12            0             0             0       12   63    0    0    1    1    1    1    0  0.87   1
      scissors            2             0             1             0             0             6             0             0        9   75    3    6  0.67 0.92  0.5 0.96 0.33 0.48 0.57
    scrambledpix          2             0             1             0             0             0             12            1       16   63    4    0  0.75   1    1  0.94 0.25 0.75 0.86
        shoe              2             0             0             0             0             1             0             10      13   67    3    2  0.77 0.97 0.83 0.96 0.23 0.69  0.8
Per target:          ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
         P                12            12            12            12            12            12            12            12
         N                84            84            84            84            84            84            84            84
         TP               6             10            7             12            12            6             12            10
         TN               69            65            68            63            63            69            63            65
Summary \ Means:     ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ 12 68.25 2.62 2.62 0.81 0.96 0.78 0.96 0.19 0.67 0.78
       CHI^2            442.67       p=2e-58
        ACC              0.78
        ACC%            78.12
     # of sets            12       ACC(i) = 0.87-0.015*i p=0.3 r=-0.33 r^2=0.11
<BLANKLINE>
Statistics computed in 1-vs-rest fashion per each target.
Abbreviations (for details see http://en.wikipedia.org/wiki/ROC_curve):
 TP : true positive (AKA hit)
 TN : true negative (AKA correct rejection)
 FP : false positive (AKA false alarm, Type I error)
 FN : false negative (AKA miss, Type II error)
 TPR: true positive rate (AKA hit rate, recall, sensitivity)
      TPR = TP / P = TP / (TP + FN)
 FPR: false positive rate (AKA false alarm rate, fall-out)
      FPR = FP / N = FP / (FP + TN)
 ACC: accuracy
      ACC = (TP + TN) / (P + N)
 SPC: specificity
      SPC = TN / (FP + TN) = 1 - FPR
 PPV: positive predictive value (AKA precision)
      PPV = TP / (TP + FP)
 NPV: negative predictive value
      NPV = TN / (TN + FN)
 FDR: false discovery rate
      FDR = FP / (FP + TP)
 MCC: Matthews Correlation Coefficient
      MCC = (TP*TN - FP*FN)/sqrt(P N P' N')
 F1 : F1 score
      F1 = 2TP / (P + P') = 2TP / (2TP + FP + FN)
 AUC: Area under (AUC) curve
 CHI^2: Chi-square of confusion matrix
 LOE(ACC): Linear Order Effect in ACC across sets
 # of sets: number of target/prediction sets which were provided
<BLANKLINE>

This output is a comprehensive summary of the performed analysis. We can
see that the confusion matrix has a strong diagonal, and confusion happens
mostly among small objects. In addition to the plain contingency table
there are also a number of useful summary statistics readily available --
including average accuracy.

Especially for multi-class datasets the matrix quickly becomes
incomprehensible. For these cases the confusion matrix can also be plotted
via its `~mvpa2.clfs.transerror.ConfusionMatrix.plot()` method. If the
confusions shall be used as input for further processing they can also be
accessed in pure matrix format:

>>> print cvte.ca.stats.matrix
[[ 6  0  3  0  0  5  0  1]
 [ 0 10  0  0  0  0  0  0]
 [ 0  0  7  0  0  0  0  0]
 [ 0  2  0 12  0  0  0  0]
 [ 0  0  0  0 12  0  0  0]
 [ 2  0  1  0  0  6  0  0]
 [ 2  0  1  0  0  0 12  1]
 [ 2  0  0  0  0  1  0 10]]

The classifier confusions are just an example of the general mechanism of
conditional attribute that is supported by many objects in PyMVPA.
