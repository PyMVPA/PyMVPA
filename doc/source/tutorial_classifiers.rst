.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. vi: set ft=rst sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. index:: Tutorial
.. _chap_tutorial_classifiers:

***********************************************
Part 4: Classifiers -- All Alike, Yet Different
***********************************************

This is already the second time that we will engage in a classification
analysis, so let's first recap what we did before in the :ref:`first tutorial
part <chap_tutorial_start>`:

>>> from tutorial_lib import *
>>> ds = get_haxby2001_data()
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

Our previous choice of the classifier was guided by the intention to
replicate the :ref:`Haxby et al. (2001) <HGF+01>`, but what if we want to
try a different algorithm? In this case a nice feature of PyMVPA comes into
play. All classifiers implement a common interface that makes them easily
exchangeable without the need to adapt any other part of the analysis code.
If, for example, we want to try the popular :mod:`support vector machine <mvpa.clfs.svm>`
(SVM) on our example dataset it looks like this:

>>> clf = LinearCSVMC()
>>> cvte = CrossValidation(clf, HalfPartitioner(attr='runtype'))
>>> cv_results = cvte(ds)
>>> np.mean(cv_results)
0.1875

Instead of k-nearest-neighbor, we create a linear SVM classifier,
internally using popular LIBSVM library (note that PyMVPA provides
additional SVM implementations). The rest of the code remains identical.
SVM with its default settings seems to perform slightly worse than the
simple kNN-classifier. We'll get back to the classifiers shortly. Let's
first look at the remaining part of this analysis.

We already know that `~mvpa.clfs.transerror.TransferError` is used to compute
the error function. So far we have used only the mean mismatch between actual
targets and classifier predictions as the error function (which is the default).
However, PyMVPA offers a number of alternative functions in the
:mod:`mvpa.misc.errorfx` module, but it is also trivial to specify custom ones.
For example, if we do not want to have error reported, but instead accuracy, we
can do that:

>>> cvte = CrossValidation(clf, HalfPartitioner(attr='runtype'),
...                        errorfx=lambda p, t: np.mean(p == t))
>>> cv_results = cvte(ds)
>>> np.mean(cv_results)
0.8125

This example reuses the SVM classifier we have create before, and
yields exactly what we expect from the previous result.

The details of the cross-validation procedure are also heavily
customizable. We have seen that a `~mvpa.datasets.splitters.Splitter` is
used to generate training and testing dataset for each cross-validation
fold. So far we have only used `~mvpa.datasets.splitters.HalfSplitter` to
divide the dataset into odd and even runs (based on our custom sample
attribute ``runtype``). However, in general it is more common to perform so
called leave-one-out cross-validation, where *one* independent part of a
dataset is selected as testing dataset, while the other parts constitute the
training dataset. This procedure is repeated till all parts have served as
the testing dataset once. In case of our dataset we could consider each of
the 12 runs as independent measurements (fMRI data doesn't allow us to
consider temporally adjacent data to be considered independent).

To run such an analysis we first need to redo our dataset preprocessing,
since in the current one we only have one sample per stimulus category for
both odd and even runs. To get a dataset with one sample per stimulus
category for each run, we need to modify the averaging step. Using what we
have learned from the :ref:`last tutorial part <chap_tutorial_mappers>` the
following code snippet should be plausible:

>>> # directory that contains the data files
>>> datapath = os.path.join(tutorial_data_path, 'data')
>>> # load the raw data
>>> attr = SampleAttributes(os.path.join(datapath, 'attributes.txt'))
>>> ds = fmri_dataset(samples=os.path.join(datapath, 'bold.nii.gz'),
...                   targets=attr.targets, chunks=attr.chunks,
...                   mask=os.path.join(datapath, 'mask_vt.nii.gz'))
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
make use of `~mvpa.datasets.splitters.NFoldSplitter`. By default it is
going to select samples from one ``chunk`` at a time:

>>> cvte = CrossValidation(clf, NFoldPartitioner,
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
<class 'mvpa.datasets.base.Dataset'>
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

Returned value is actually a `~mvpa.datasets.base.Dataset` with the
results for all cross-validation folds. Since our error function computes
only a single scalar value for each fold the dataset only contain a single
feature (in this case the accuracy), and a sample per each fold. Moreover,
the dataset also offers a sample attribute that show which particular set
of chunks formed the training and testing set per fold.

>>> print cv_results.sa.cv_fold
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
important for an interpretation of results. In our case we analyze a
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
characterize classifier's performance based on that. From this angle a
classification analysis is hardly any different from a psychological
experiment where a human observer performs a detection task, hence the same
analysis procedures can be applied here as well.

.. _contingency table: http://en.wikipedia.org/wiki/Contingency_table
.. _cross tabulation: http://en.wikipedia.org/wiki/Cross_tabulation
.. _signal detection theory: http://en.wikipedia.org/wiki/Detection_theory

PyMVPA provides convenient access to :term:`confusion matrices <confusion matrix>`, i.e.
contingency tables of targets vs. actual predictions.  However, to prevent
wasting CPU-time and memory they are not computed by default, but instead
have to be enabled explicitly. Optional analysis results like this are
available in a dedicated collection of :term:`conditional attribute`\ s --
analogous to ``sa`` and ``fa`` in datasets, it is named ``ca``. Let's see
how it works:

>>> cvte = CrossValidation(clf, NFoldPartitioner,
...                        errorfx=lambda p, t: np.mean(p == t),
...                        enable_ca=['stats'])
>>> cv_results = cvte(ds)

Via the ``enable_ca`` argument we triggered computing confusion tables for
all cross-validation folds, but otherwise there is no change in the code.
Afterwards the aggregated confusion for the whole cross-validation
procedure is available in the ``ca`` collection. Let's take a look (note
that in the printed manual the output is truncated due to page width
constraints -- please refer to the HTML-based version full the full matrix).

>>> print cvte.ca.stats.as_string(description=True)
----------.
predictions\targets     bottle         cat          chair          face         house        scissors    scrambledpix      shoe
            `------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ P'   N'   FP   FN   PPV  NPV  TPR  SPC  FDR  MCC
       bottle             6             0             3             0             0             5             0             1       15   75    9    6   0.4 0.92  0.5 0.88  0.6 0.34
        cat               0             10            0             0             0             0             0             0       10   67    0    2    1  0.97 0.83   1    0  0.79
       chair              0             0             7             0             0             0             0             0        7   73    0    5    1  0.93 0.58   1    0  0.66
        face              0             2             0             12            0             0             0             0       14   63    2    0  0.86   1    1  0.97 0.14  0.8
       house              0             0             0             0             12            0             0             0       12   63    0    0    1    1    1    1    0  0.87
      scissors            2             0             1             0             0             6             0             0        9   75    3    6  0.67 0.92  0.5 0.96 0.33 0.48
    scrambledpix          2             0             1             0             0             0             12            1       16   63    4    0  0.75   1    1  0.94 0.25 0.75
        shoe              2             0             0             0             0             1             0             10      13   67    3    2  0.77 0.97 0.83 0.96 0.23 0.69
Per target:          ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
         P                12            12            12            12            12            12            12            12
         N                84            84            84            84            84            84            84            84
         TP               6             10            7             12            12            6             12            10
         TN               69            65            68            63            63            69            63            65
Summary \ Means:     ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------ 12 68.25 2.62 2.62 0.81 0.96 0.78 0.96 0.19 0.67
       CHI^2            442.67          p:          2e-58
        ACC              0.78
        ACC%            78.12
     # of sets            12
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
 AUC: Area under (AUC) curve
 CHI^2: Chi-square of confusion matrix
 # of sets: number of target/prediction sets which were provided
<BLANKLINE>

This output is a comprehensive summary of the performed analysis. We can
see that the confusion matrix has a strong diagonal, and confusion happens
mostly among small objects. In addition to the plain contingency table
there are also a number of useful summary statistics readily available --
including average accuracy.

Especially for multi-class datasets the matrix quickly becomes
incomprehensible. For these cases the confusion matrix can also be plotted
via its `~mvpa.clfs.transerror.ConfusionMatrix.plot()` method. If the
confusions shall be used as input for further processing they can also be
accessed in pure matrix format:

>>> print cvte.ca.tats.matrix
[[ 6  0  3  0  0  5  0  1]
 [ 0 10  0  0  0  0  0  0]
 [ 0  0  7  0  0  0  0  0]
 [ 0  2  0 12  0  0  0  0]
 [ 0  0  0  0 12  0  0  0]
 [ 2  0  1  0  0  6  0  0]
 [ 2  0  1  0  0  0 12  1]
 [ 2  0  0  0  0  1  0 10]]

The classifier confusions are just an example of the general mechanism of
conditional attribute that is supported by many objects in PyMVPA. The
docstring of `~mvpa.algorithms.cvtranserror.CrossValidatedTransferError`
and others lists more information that can be enabled on demand.


Meta-Classifiers To Make Complex Stuff Simple
=============================================

We just saw that it is possible to encapsulate a whole cross-validation
analysis into a single object that can be called with any dataset to
produce the desired results. We also saw that despite this encapsulation we
can still get a fair amount of information about the performed analysis.
However, what happens if we want to do some further processing of the data
**within** the cross-validation analysis. That seems to be difficult, since
we feed a whole dataset into the analysis, and only internally it get split
into the respective pieces.

Of course there is a solution to this problem -- a :term:`meta-classifier`.
This is a classifier that doesn't implement a classification algorithm on
its own, but uses another classifier to do the actual work. In addition,
the meta-classifier adds another processing step that is performed before
the actual :term:`base-classifier` sees the data.

An example of such meta-classifier is `~mvpa.clfs.meta.MappedClassifier`.
Its purpose is simple: Apply a mapper to both training and testing data
before it is passed on to the internal base-classifier. With this technique
it is possible to implement arbitrary pre-processing within a
cross-validation analysis. Suppose we want to perform the classification
not on voxel intensities themselves, but on the same samples in the space
spanned by the singular vectors of the training data, it would look like this:

>>> baseclf = LinearCSVMC()
>>> metaclf = MappedClassifier(baseclf, SVDMapper())
>>> cvte = CrossValidation(metaclf, NFoldPartitioner())
>>> cv_results = cvte(ds)
>>> print np.mean(cv_results)
0.15625

First we notice that little has been changed in the code and the results --
the error is slightly reduced, but still comparable. The critical line is
the second, where we create the `~mvpa.clfs.meta.MappedClassifier` from the
SVM classifier instance, and a `~mvpa.mappers.svd.SVDMapper` that
implements `singular value decomposition`_ as a mapper.

.. exercise::

   What might be the reasons for the error decrease in comparison to the
   results on the dataset with voxel intensities?

.. _singular value decomposition: http://en.wikipedia.org/wiki/Singular_value_decomposition

We know that mappers can be combined into complex processing pipelines, and
since `~mvpa.clfs.meta.MappedClassifier` takes any mapper as argument, we
can implement arbitrary preprocessing steps within the cross-validation
procedure. Let's say we have heard rumors that only the first two dimensions
of the space spanned by the SVD vectors cover the "interesting" variance
and the rest is noise. We can easily check that with an appropriate mapper:

>>> mapper = ChainMapper([SVDMapper(), FeatureSliceMapper(slice(None, 2))])
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
>>> mapper = ChainMapper([SVDMapper(), FeatureSliceMapper(slice(None, 2))])
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

In this tutorial part we took a look at classifiers. We have seen that
regardless of the actual algorithm all classifiers are implementing the same
interface. Because of that they can be replaced by another classifier without
having to change any other part of the analysis code. Moreover, we have seen
that it is possible to enable and access optional information that is offered
by particular parts of the processing pipeline.

However, we still have done little to address one of the major questions in
neuroscience research, that is: Where does the information come from? One
possible approach to this question is the topic of the :ref:`next tutorial part
<chap_tutorial_searchlight>`.

.. Think about adding a demo of the classifiers warehouse.
  .. exercise::
     Try doing the Z-Scoring before computing the mean samples per category.
     What happens to the generalization performance of the classifier?
     ANSWER: It becomes 100%!


.. only:: html

  References
  ==========

  .. autosummary::
     :toctree: generated

     ~mvpa.clfs.base.Classifier
