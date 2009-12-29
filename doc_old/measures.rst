.. -*- mode: rst; fill-column: 78 -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


.. index:: measure, sensitivity
.. _chap_measures:

********
Measures
********


PyMVPA provides a number of useful measures. The vast majority of
them are dedicated to feature selection. To increase analysis
flexibility, PyMVPA distinguishes two parts of a feature selection
procedure.

First, the impact of each individual feature on a classification has to be
determined.  The resulting map reflects the sensitivities of all features with
respect to a certain decision and, therefore, algorithms generating these maps
are summarized as :class:`~mvpa.measures.base.Sensitivity` in PyMVPA.

.. index:: feature selection

Second, once the feature sensitivities are known, they can be used as
criteria for feature selection. However, possible selection strategies
range from very simple *Go with the 10% best features* to more
complicated algorithms like :ref:`recursive_feature_elimination`.
Because :ref:`sensitivity_measures` and selections strategies can be
arbitrarily combined, PyMVPA offers a quite flexible framework for feature
selection.

.. index:: processing object

Similar to dataset splitters, all PyMVPA algorithms are implemented and behave
like :term:`processing object`\ s. To recap, this means that they are
instantiated by passing all relevant arguments to the constructor. Once
created, they can be used multiple times by calling them with different
datasets.

.. Again general overview first. What is a `SensitivityAnalyzer`, what is the
   difference between a `FeatureSelection` and an `ElementSelector`.
   Finally more detailed note and references for each larger algorithm.


.. index:: sensitivity
.. _sensitivity_measures:

Sensitivity Measures
====================

It was already mentioned that a :class:`~mvpa.measures.base.Sensitivity`
computes a featurewise score that indicates how much interesting signal each
feature contains -- hoping that this score somehow correlates with the impact
of the features on a classifier's decision for a certain problem.

Every sensitivity analyzer object computes a one-dimensional array with the
respective score for every feature, when called with a
:class:`~mvpa.datasets.base.Dataset`. Due to this common behavior all
:class:`~mvpa.measures.base.Sensitivity` types are interchangeable and can be
combined with any other algorithm requiring a sensitivity analyzer.

By convention higher sensitivity values indicate more interesting features.

There are two types of sensitivity analyzers in PyMVPA. Basic sensitivity
analyzers directly compute a score from a Dataset. Meta sensitivity analyzers
on the other hand utilize another sensitivity analyzer to compute their
sensitivity maps.


Basic Sensitivity (and related Measures)
----------------------------------------

.. index:: anova, F-score, univariate, measure
.. _anova:

ANOVA
^^^^^

The :class:`~mvpa.measures.anova.OneWayAnova` class provides a simple (and fast)
univariate measure, that can be used for feature selection, although it is not
a proper sensitivity measure. For each feature an individual F-score is
computed as the fraction of between and within group variances. Groups are
defined by samples with unique labels.

Higher F-scores indicate higher sensitivities, as with all other sensitivity
analyzers.



.. index:: classifier weights, weights, SVM, measure

Linear SVM Weights
^^^^^^^^^^^^^^^^^^

The featurewise weights of a trained support vector machine are another
possible sensitivity measure.  The
:class:`mvpa.clfs.libsvmc.sens.LinearSVMWeights` and
:class:`mvpa.clfs.sg.sens.LinearSVMWeights` classes can internally train all
types of *linear* support vector machines and report those weights.

In contrast to the F-scores computed by an ANOVA, the weights can be positive
or negative, with both extremes indicating higher sensitivities. To deal with
this property all subclasses of :class:`~mvpa.measures.base.DatasetMeasure`
support a `transformer` arguments in the constructor. A transformer is a functor
that is finally called with the computed sensitivity map. PyMVPA already comes
with some convenience functors which can be used for this purpose (see
:mod:`~mvpa.misc.transformers`).

 >>> from mvpa.misc.data_generators import normalFeatureDataset
 >>> from mvpa.clfs.svm import LinearCSVMC
 >>> from mvpa.misc.transformers import Absolute
 >>>
 >>> ds = normalFeatureDataset()
 >>> ds
 <Dataset / float64 100 x 4 uniq: 2 labels 5 chunks labels_mapped>
 >>>
 >>> clf = LinearCSVMC()
 >>> sensana = clf.getSensitivityAnalyzer()
 >>> sens = sensana(ds)
 >>> sens.shape
 (4,)
 >>> (sens < 0).any()
 True
 >>> sensana_abs = clf.getSensitivityAnalyzer(transformer=Absolute)
 >>> (sensana_abs(ds) < 0).any()
 False

Above example shows how to use an existing classifier instance to report
sensitivity values (a linear SVM in this case). The computed sensitivity vector
contains one element for each feature in the dataset.
:mod:`~mvpa.misc.transformers` can be used to post-process the sensitivity
scores, e.g. reporting absolute values for feature selection purposes, instead
of raw sensitivities.

.. note::

  The `SVMWeights` classes *cannot* extract reasonable weights from non-linear
  SVMs (e.g. with RBF kernels).



Other linear Classifier Weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Any linear classifier in PyMVPA can report its weights. The procedure is
identical for all of them. As outlined in the example using linear SVM weights,
simply call :meth:`~mvpa.clfs.base.Classifier.getSensitivityAnalyzer` on a
classifier instance and you'll get an appropriate
:class:`~mvpa.measures.base.Sensitivity` object. Additionally, it is possible
to force (re)training of the underlying classifier or simply report the weights
computed during a previous training run.

Examples of other classifier-based linear sensitivity analyzers are:
:class:`~mvpa.clfs.smlr.SMLRWeights` and
:class:`~mvpa.clfs.gpr.GPRLinearWeights`.


.. index:: noise perturbation, measure
.. _noise_perturbation:

Noise Perturbation
^^^^^^^^^^^^^^^^^^

Noise perturbation is a generic approach to determine feature sensitivity.  The
sensitivity analyzer
:class:`~mvpa.measures.noiseperturbation.NoisePerturbationSensitivity`)
computes a scalar :class:`~mvpa.measures.base.DatasetMeasure` using the
original dataset. Afterwards, for each single feature a noise pattern is added
to the respective feature and the dataset measure is recomputed. The
sensitivity of each feature is the difference between the dataset measure of
the original dataset and the one with added noise. The reasoning behind this
algorithm is that adding noise to *important* features will impair a dataset
measure like cross-validated classifier transfer error. However, adding noise
to a feature that already only contains noise, will not change such a measure.

Depending on the used scalar :class:`~mvpa.measures.base.DatasetMeasure` using
the sensitivity analyzer might be really CPU-intensive! Also depending on the
measure, it might be necessary to use appropriate
:mod:`~mvpa.misc.transformers` (see :mod:`~mvpa.misc.transformers` constructor
arguments) to ensure that higher values represent higher sensitivities.


.. index:: meta measures

Meta Sensitivity Measures
-------------------------

Meta Sensitivity Measures are FeaturewiseDatasetMeasures that internally use
one of the `Basic Sensitivity (and related Measures)`_ to compute their
sensitivity scores.


.. index:: splitting measures, measure

Splitting Measures
^^^^^^^^^^^^^^^^^^

The SplittingFeaturewiseMeasure uses a
:class:`~mvpa.datasets.splitters.Splitter` to generate dataset splits.  A
FeaturewiseDatasetMeasure is then used to compute sensitivity maps for all
these dataset splits. At the end a `combiner` function is called with all
sensitivity maps to produce the final sensitivity map. By default the mean
sensitivity maps across all splits is computed.

.. _SplitFeaturewiseMeasure: api/mvpa.measures.splitmeasure.SplitFeaturewiseMeasure-class.html
