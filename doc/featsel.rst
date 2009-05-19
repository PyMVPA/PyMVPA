.. -*- mode: rst; fill-column: 78 -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


.. index:: feature selection
.. _chap_featsel:

*****************
Feature Selection
*****************

  *This section has been contributed by James M. Hughes.*

It is often the case in machine learning problems that we wish to reduce a
feature space of high dimensionality into something more manageable by
selecting only those features that contribute most to classification
performance.  Feature selection methods attempt to achieve this goal in an
algorithmic fashion.

.. index:: FeatureSelectionClassifier

PyMVPA's flexible framework allows various feature selection methods to take
place within a small block of code.  :class:`~mvpa.clfs.meta.FeatureSelectionClassifier` extends the
basic classifier framework to allow for the use of arbitrary methods of feature
selection according to whatever ranking metric, feature selection criteria, and
stopping criterion the user chooses for a given application.  Examples of the
code/classification algorithms presented here can be found in
`mvpa/clfs/warehouse.py`_.

More formally, a :class:`~mvpa.clfs.meta.FeatureSelectionClassifier` is a meta-classifier.  That is, it
is not a classifier itself -- it can take any *slave* :class:`~mvpa.clfs.base.Classifier`, perform some
feature selection in advance, select those features, and then train the
provided *slave* :class:`~mvpa.clfs.base.Classifier` on those features.  Externally, however, it looks
like a :class:`~mvpa.clfs.base.Classifier`, in that it fulfills the specialization of the Classifier
base class.  The following are the relevant arguments to the constructor of
such a :class:`~mvpa.clfs.base.Classifier`:

`clf`: :class:`~mvpa.clfs.base.Classifier`
  classifier based on which mask classifiers is created

`feature_selection`: FeatureSelection_
  whatever feature selection is considered best

`testdataset`: :class:`~mvpa.datasets.base.Dataset` (optional)
  dataset which would be given on call to feature_selection

.. index:: FeatureSelection

Let us turn out attention to the second argument, FeatureSelection_. As noted
above, this feature selection can be arbitrary and should be chosen
appropriately for the task at hand.  For example, we could perform a one-way
ANOVA statistic to select features, then keep only the most important 5% of
them.  It is crucial to note that, in PyMVPA, the way in which features are
selected (in this example by keeping only 5% of them) is wholly independent of
the way features are ranked (in this example, by using a one-way ANOVA).
Feature selection using this method could be accomplished using the following
code (from `mvpa/clfs/warehouse.py`_):

  >>> from mvpa.suite import *
  >>> FeatureSelection = SensitivityBasedFeatureSelection(
  ...     OneWayAnova(),
  ...     FractionTailSelector(0.05, mode='select', tail='upper'))

A more interesting analysis is one in which we use the weights (hyperplane
coefficients) to rank features.  This allows us to use the same classifier to
train the selected features as we used to select them:

.. here we'll put the warehouse.py example of linear svm weights from yarik's
   email

  >>> sample_linear_svm = clfswh['linear', 'svm'][0]
  >>> clf = \
  ...  FeatureSelectionClassifier(
  ...      sample_linear_svm,
  ...      SensitivityBasedFeatureSelection(
  ...         sample_linear_svm.getSensitivityAnalyzer(transformer=Absolute),
  ...         FractionTailSelector(0.05, mode='select', tail='upper')),
  ...      descr="LinSVM on 5%(SVM)")

It bears mentioning at this point that caution must be exercised when selecting
features.  The process of feature selection must be performed on an independent
training dataset:  it is not possible to select features using the entire
dataset, re-train a classifier on a subset of the original data (but using only
the selected features) and then test on a held-out testing dataset.  This
results in an obvious positive bias in classification performance.  PyMVPA
allows for easy dataset splitting, however, so creating independent training
and testing datasets is easily accomplished, for instance using an
:class:`~mvpa.datasets.splitters.NFoldSplitter`, :class:`~mvpa.datasets.splitters.OddEvenSplitter`, etc.

.. fill in end of last paragraph with suggestions for how to take in an entire
   original dataset and split it:  should we just do a cross-validated outer
   loop that uses multiple training/testing splits and does RFE on each of
   these splits?



.. index:: recursive feature selection, RFE
.. _recursive_feature_elimination:

Recursive Feature Elimination
=============================

Recursive feature elimination
(RFE_, applied to fMRI data in (:ref:`Hanson et al., 2008 <HH08>`))
is a technique that falls under the larger
umbrella of feature selection. Recursive feature elimination specifically
attempts to reduce the number of selected features used for classification in
the following way:

* A classifier is trained on a subset of the data and features are ranked
  according to an arbitrary metric.

* Some amount of those features is either selected or discarded according to a
  pre-selected rule.

* The classifier is retrained and features are once again ranked; this process
  continues until some criterion determined \textit{a priori} (such as
  classification error) is reached.

* One or more classifiers trained only on the final set of selected features
  are used on a generalization dataset and performance is calculated.

PyMVPA's flexible framework allows each of these steps to take place within a
small block of code. To actually perform recursive feature elimination, we
consider two separate analysis scenarios that deal with a pre-selected training
dataset:

* We split the training dataset into an arbitrary number of independent
  datasets and perform RFE on each of these; the sensitivity analysis of
  features is performed independently for each split and features are selected
  based on those independent measures.

* We split the training dataset into an arbitrary number of independent
  datasets (as before), but we average the feature sensitivities and select
  which features to prune/select based on that one average measure.

.. index:: SplitClassifier

We will concentrate on the second approach.  The following code can be used to
perform such an analysis:

  >>> rfesvm_split = SplitClassifier(LinearCSVMC())
  >>> clf = \
  ...  FeatureSelectionClassifier(
  ...   clf = LinearCSVMC(),
  ...   # on features selected via RFE
  ...   feature_selection = RFE(
  ...       # based on sensitivity of a clf which does splitting internally
  ...       sensitivity_analyzer=rfesvm_split.getSensitivityAnalyzer(),
  ...       transfer_error=ConfusionBasedError(
  ...          rfesvm_split,
  ...          confusion_state="confusion"),
  ...          # and whose internal error we use
  ...       feature_selector=FractionTailSelector(
  ...                          0.2, mode='discard', tail='lower'),
  ...                          # remove 20% of features at each step
  ...       update_sensitivity=True),
  ...       # update sensitivity at each step
  ...   descr='LinSVM+RFE(splits_avg)' )

The code above introduces the :class:`~mvpa.clfs.meta.SplitClassifier`, which in this case is yet
another *meta-classifier* that takes in a :class:`~mvpa.clfs.base.Classifier` (in this case a
LinearCSVMC_) and an arbitrary :class:`~mvpa.datasets.splitters.Splitter` object, so that the dataset can be
split in whatever way the user desires.  Prior to training, the
:class:`~mvpa.clfs.meta.SplitClassifier` splits the training dataset, dedicates a separate classifier
to each split, trains each on the training part of the split, and then computes
transfer error on the testing part of the split. If a :class:`~mvpa.clfs.meta.SplitClassifier` instance
is later on asked to *predict* some new data, it uses (by default) the
MaximalVote_ strategy to derive an answer.  A summary about the performance of
a :class:`~mvpa.clfs.meta.SplitClassifier` internally on each split of the training dataset is
available by accessing the `confusion` state variable.

To summarize somewhat, RFE_ is just one method of feature selection, so we use a
:class:`~mvpa.clfs.meta.FeatureSelectionClassifier` to facilitate this.  To parameterize the RFE
process, we refer above to the following:

`sensitivity_analyzer`
  in this case just the default from a linear C-SVM (the SVM weights), taken as
  an average over all splits (in accordance with scenario 2 as above)

`transfer_error`
  confusion-based error that relies on the confusion matrices computed during
  splitting of the dataset by the :class:`~mvpa.clfs.meta.SplitClassifier`; this is used to provide a
  value that can be compared against a stopping criterion to stop eliminating
  features

`feature_selector`
  in this example we simply discard the 20% of features deemed least important

`update_sensitivity`
  true to retrain the classifiers each time we eliminate features; should be
  false if a non-classifier-based sensitivity measure (such as one-way ANOVA)
  is used

As has been shown, recursive feature elimination is an easy-to-implement,
flexible, and powerful tool within the PyMVPA framework.  Various ranking
methods for selecting features have been discussed.  Additionally, several
analysis scenarios have been presented, along with enough requisite knowledge
that the user can plug in whatever classifiers, error metrics, or sensitivity
measures are most appropriate for the task at hand.

.. _RFE: api/mvpa.featsel.rfe.RFE-class.html

.. _MaximalVote: api/mvpa.clfs.meta.MaximalVote-class.html
.. _FeatureSelection: api/mvpa.featsel.base.FeatureSelection-class.html
.. _LinearCSVMC: api/mvpa.clfs.svm.LinearCSVMC-class.html
.. _mvpa/clfs/warehouse.py: api/mvpa.clfs.warehouse-pysrc.html


.. index:: incremental feature search, IFS
.. _incremental_feature_search:

Incremental Feature Search
==========================

IFS_

(to be written)

.. _IFS: api/mvpa.featsel.ifs.IFS-class.html

.. What are the practical differences (besides speed) between RFE and IFS?

