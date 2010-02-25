.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. index:: Tutorial
.. _chap_tutorial_sensitivity:

*********************************************************
Part 6: Looking Without Searching -- Sensitivity Analysis
*********************************************************

In the :ref:`previous tutorial part <chap_tutorial_searchlight>` we made a
first attempt to localize information in the brain that is relevant to a
particular classification analyses. While we were realatively successful,
we experienced some problem and also had to wait quite a bit. Here we want
to look at another approach to localization. To get started we preprocess
the data as we have done before and perform volume averaging to get a
single sample per stimulus category and original experiment session.

>>> from tutorial_lib import *
>>> ds = get_raw_haxby2001_data(roi='brain')
>>> print ds.shape
(1452, 39912)
>>> # pre-process
>>> poly_detrend(ds, polyord=1, chunks_attr='chunks')
>>> zscore(ds, param_est=('targets', ['rest']))
>>> ds = ds[ds.sa.targets != 'rest']
>>> # average
>>> run_averager = mean_group_sample(['targets', 'chunks'])
>>> ds = ds.get_mapped(run_averager)
>>> print ds.shape
(96, 39912)

A searchlight analysis on this dataset would look exactly as we have seen
:ref:`before <chap_tutorial_searchlight>`, but it would take a bit longer
due to a higher number of samples. The error map that is the result of a
searchlight analysis only offers an approximate localization. First it is
smeared by the overlapping spheres and second the sphere shaped ROIs
probably do not reflect the true shape and extent of functional subregions
in the brain. Therefore it mixes and matches things that might not belong
together. It would be much nicer if we would be able to obtain a
per-feature measure, where each value can really be attributed to the
respective feature and not an area surrounding it.

It's A Kind Of Magic
--------------------

One way to get such a measure is to inspect the classifier itself. Each
classifier creates a model to map from the training data onto the
respective target values. In this model classifiers typically associate
some sort of weight with each feature that is an indication of its impact
on the classifiers decision. How we can get this information from a
classifier will be the topic of this tutorial.

However, if we want to inspect a trained classifier, we first have to train
one. But hey, we have a full brain dataset here with almost 40k features.
Will we be able to do that? Well, let's try (and hope that there is still a
waranty on the machine you are running this on...).

We will use a simple cross-validation procedure with a linear support
vector machine and we want a confusion matrix:

>>> clf = LinearCSVMC()
>>> cvte = CrossValidatedTransferError(
...             TransferError(clf),
...             splitter=NFoldSplitter(),
...             enable_ca=['confusion'])

Ready, set, go!

>>> results = cvte(ds)

That was surprisingly quick, wasn't it? But was it any good?

>>> print N.round(cvte.ca.confusion.stats['ACC%'], 1)
26.0
>>> print cvte.ca.confusion.matrix
[[1 1 2 3 0 1 1 1]
 [1 2 2 0 2 2 3 1]
 [5 3 3 0 4 4 0 2]
 [3 2 0 5 0 0 0 1]
 [0 3 1 0 3 2 0 0]
 [0 0 0 0 0 0 1 0]
 [0 1 4 3 2 1 7 3]
 [2 0 0 1 1 2 0 4]]

Well, the accuracy is not exactly chance, but the confusion matrix doesn't
seem to have any visible diagonal. It looks like, although we can easily
train a support vector machine on the full brain dataset, it doesn't learn
anything useful. At least we are in the lucky situation to already know
that there is some signal in the data, hence we can attribute this failure
to the classifier. In most situations it would be as likely that there is
actually no signal in the data...

Often people claim that classification performance improves with :term:`feature
selection`. If we can reduce the dataset to the important ones the
classifier wouldn't have to deal with all the noise anymore. A simple
approach would be to compute an full-brain ANOVA and only go with the
voxels that show some level of variance between category.From the
:ref:`previous tutorial part <chap_tutorial_searchlight>` we know how to
compute the desired F-score and could use them to manually select features
with some threshold. However, PyMVPA offers a more convenient way --
feature selectors:

>>> fsel = SensitivityBasedFeatureSelection(
...            OneWayAnova(),
...            FixedNElementTailSelector(500, mode='select', tail='upper'))

The code snippet above configures such selector. It uses an ANOVA measure
to select those features that correspond to the 500 highest F-scores. There
are a lot more ways to perform the selection, but we will go with this one
for now. The :class:`~mvpa.featsel.base.SensitivityBasedFeatureSelection`
instance is yet another :term:`processing object` that can be called with a
dataset to perform the feature selection:

.. Put slicing logic from Splitters also in these objects
.. refactor them to return just one dataset

>>> ds_p = fsel(ds)[0]
>>> print ds_p.shape
(96, 500)

This is the dataset we wanted, so we can rerun the cross-validation and see
if it helped:

>>> results = cvte(ds_p)
>>> print N.round(cvte.ca.confusion.stats['ACC%'], 1)
79.2
>>> print cvte.ca.confusion.matrix
[[ 5  0  3  0  0  3  0  2]
 [ 0 11  0  0  0  0  0  0]
 [ 0  0  7  0  0  1  0  0]
 [ 2  1  0 12  0  0  0  0]
 [ 0  0  0  0 12  0  0  0]
 [ 2  0  1  0  0  8  0  0]
 [ 0  0  1  0  0  0 12  1]
 [ 3  0  0  0  0  0  0  9]]

Yes! We did it. Almost 80% correct classification for an 8-way
classification and the confusion matrix has a strong diagonal. Apparently,
the ANOVA-selected features were the right ones.

.. exercise::

  If you are not yet screaming and or started composing an email to the
  PyMVPA mailing list pointing to a major problem in the tutorial, you need
  to reconsider what we have just done. Why is this wrong?

Let's repeat this analysis on a subset of the data. We select only ``bottle``
and ``shoe`` samples. In the analysis we just did they are relatively often
confused by the classifier. Let's see how the full brain SVM performs on
this binary problem

>>> bin_demo = ds[N.array([i in ['bottle', 'shoe'] for i in ds.sa.targets])]
>>> results = cvte(bin_demo)
>>> print N.round(cvte.ca.confusion.stats['ACC%'], 1)
62.5

Not much, but that doesn't surprise. Let's see what effect our ANOVA-based
feature selection has

>>> bin_demo_p = fsel(bin_demo)[0]
>>> results = cvte(bin_demo_p)
>>> print cvte.ca.confusion.stats["ACC%"]
100.0

Wow, that is a jump. Perfect classification performance, even though the
same categories couldn't be distinguished by the same classifier, when
trained on all eight categories. I guess, it is obvious that our way of
selecting features is somewhat fishy -- if not illegal. The ANOVA measure
uses the full dataset to compute the F-score, hence it determines which
feature show category differences in the whole dataset, including our
suposed-to-be independent testing data. Once we have found these
differences, we are trying to rediscover them with a classifier. That we
are able to do that is not only surprising. Moreover, the prediction
accuracy and potentially also the created model are completely meaningless.

Thanks For The Fish
-------------------

To implement an ANOVA-based feature selection properly we have to do it on
the training dataset **only**. The PyMVPA way of doing this is via a
:class:`~mvpa.clfs.base.FeatureSelectionClassifier`:

>>> fclf = FeatureSelectionClassifier(clf, fsel)

This is a :term:`meta-classifier` and t just needs two things: A basic
classifier to do the actual classification work and a feature selection
object. We can simple re-use the object instances we already had. now we
got a meta-classifier that can be used just as any other classifier. Most
importantly we can plug it into a cross-validation procedure (almost
identical as in the beginning).

>>> cvte = CrossValidatedTransferError(
...             TransferError(fclf),
...             splitter=NFoldSplitter(),
...             enable_ca=['confusion'])
>>> results = cvte(bin_demo)
>>> print N.round(cvte.ca.confusion.stats['ACC%'], 1)
70.8

This is a lot worse and a lot closer to the truth -- or a so-called
:term:`unbiased estimate` of the generalizability of the classifier model.
We can now also run this improved procedure on our original 8-category
dataset.

>>> results = cvte(ds)
>>> print N.round(cvte.ca.confusion.stats['ACC%'], 1)
78.1
>>> print cvte.ca.confusion.matrix
[[ 5  0  2  0  0  4  0  2]
 [ 0 10  0  0  0  0  0  0]
 [ 0  0  8  0  0  1  0  0]
 [ 2  2  0 12  0  0  0  0]
 [ 0  0  0  0 12  0  0  0]
 [ 1  0  1  0  0  7  0  0]
 [ 0  0  1  0  0  0 12  1]
 [ 4  0  0  0  0  0  0  9]]

That is still a respectable accuracy for an 8-way classification and the
confusion table also confirms this.


Dissect The Classifier
----------------------

But now back to our original goal: getting the classifier's oppinion about
the importance of features in the dataset. With the approach we have used
above, the classifier is trained on 500 features. We can only have its
oppinion about those. Although this is just few times larger than a typical
searchlight sphere, we already have lifted the spatial constraint of
searchlights -- these features can come from all over the brain.

However, we still want to judge more feature, so we are changing the
feature selection to retain more.

>>> fsel = SensitivityBasedFeatureSelection(
...            OneWayAnova(),
...            FractionTailSelector(0.05, mode='select', tail='upper'))
>>> fclf = FeatureSelectionClassifier(clf, fsel)
>>> cvte = CrossValidatedTransferError(
...             TransferError(fclf),
...             splitter=NFoldSplitter(),
...             enable_ca=['confusion'])
>>> results = cvte(ds)
>>> print N.round(cvte.ca.confusion.stats['ACC%'], 1)
70.8

A drop of 8% in accuracy on about 4 times the number of features. This time
we asked for the top 5% of F-scores.

But how do we get the weight, finally? In PyMVPA (almost) each classifier
is accompanied with a so-called :term:`sensitivity analyzer`. This is an
object that know how to get them from a particular classifier type (since
each classification algorithm hides them in different places). To create
this *analyzer* we can simply ask the classifier to do it:

>>> sensana = fclf.get_sensitivity_analyzer()
<class 'mvpa.measures.base.FeatureSelectionClassifierSensitivityAnalyzer'>

As you can see, this even works for our meta-classifier. And again this
analyzer is a :term:`processing object` that returns the desired weight
when called with a dataset.

>>> sens = sensana(ds)
>>> type(sens)
<class 'mvpa.datasets.base.Dataset'>
>>> print sens.shape
(28, 39912)


in CV

SplitFeaturewiseDatasetMeasure(
            NFoldSplitter(),
            fclf.get_sensitivity_analyzer(postproc=absolute_features()))
 

* post-processing with mappers

meaning of sensitivities depends on the actual classifier




.. only:: html

  References
  ==========

  .. autosummary::
     :toctree: generated

     ~mvpa.measures.base.Sensitivity
