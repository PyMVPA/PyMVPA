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
we experienced some problems and also had to wait quite a bit. Here we want
to look at another approach to localization. To get started, we pre-process
the data as we have done before and perform volume averaging to get a
single sample per stimulus category and original experiment session.

>>> from tutorial_lib import *
>>> # alt: `ds = load_tutorial_results('ds_haxby2001_blkavg_brain')`
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

.. h5save('results/ds_haxby2001_blkavg_brain.hdf5', ds)

A searchlight analysis on this dataset would look exactly as we have seen
:ref:`before <chap_tutorial_searchlight>`, but it would take a bit longer
due to a higher number of samples. The error map that is the result of a
searchlight analysis only offers an approximate localization. First, it is
smeared by the overlapping spheres and second the sphere-shaped ROIs
probably do not reflect the true shape and extent of functional subregions
in the brain. Therefore, it mixes and matches things that might not belong
together. It would be much nicer if we would be able to obtain a
per-feature measure, where each value can really be attributed to the
respective feature and not just to an area surrounding it.

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
>>> cvte = CrossValidation(clf, NFoldPartitioner(),
...                        enable_ca=['confusion'])

Ready, set, go!

>>> results = cvte(ds)

That was surprisingly quick, wasn't it? But was it any good?

>>> print np.round(cvte.ca.confusion.stats['ACC%'], 1)
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
selection`. If we can reduce the dataset to the important ones, the
classifier wouldn't have to deal with all the noise anymore. A simple
approach would be to compute an full-brain ANOVA and only go with the
voxels that show some level of variance between categories. From the
:ref:`previous tutorial part <chap_tutorial_searchlight>` we know how to
compute the desired F-scores and could use them to manually select features
with some threshold. However, PyMVPA offers a more convenient way --
feature selectors:

>>> fsel = SensitivityBasedFeatureSelection(
...            OneWayAnova(),
...            FixedNElementTailSelector(500, mode='select', tail='upper'))

The code snippet above configures such a selector. It uses an ANOVA measure
to select those features that correspond to the 500 highest F-scores. There
are a lot more ways to perform the selection, but we will go with this one
for now. The :class:`~mvpa.featsel.base.SensitivityBasedFeatureSelection`
instance is yet another :term:`processing object` that can be called with a
dataset to perform the feature selection:

.. Put slicing logic from Splitters also in these objects
.. refactor them to return just one dataset

>>> ds_p = fsel(ds)
>>> print ds_p.shape
(96, 500)

This is the dataset we wanted, so we can rerun the cross-validation and see
if it helped:

>>> results = cvte(ds_p)
>>> print np.round(cvte.ca.confusion.stats['ACC%'], 1)
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
and ``shoe`` samples. In the analysis we just did, they are relatively often
confused by the classifier. Let's see how the full brain SVM performs on
this binary problem

>>> bin_demo = ds[np.array([i in ['bottle', 'shoe'] for i in ds.sa.targets])]
>>> results = cvte(bin_demo)
>>> print np.round(cvte.ca.confusion.stats['ACC%'], 1)
62.5

Not much, but that doesn't surprise. Let's see what effect our ANOVA-based
feature selection has

>>> bin_demo_p = fsel(bin_demo)
>>> results = cvte(bin_demo_p)
>>> print cvte.ca.confusion.stats["ACC%"]
100.0

Wow, that is a jump. Perfect classification performance, even though the
same categories couldn't be distinguished by the same classifier, when
trained on all eight categories. I guess, it is obvious that our way of
selecting features is somewhat fishy -- if not illegal. The ANOVA measure
uses the full dataset to compute the F-scores, hence it determines which
features show category differences in the whole dataset, including our
suposed-to-be independent testing data. Once we have found these
differences, we are trying to rediscover them with a classifier. That we
are able to do that is not only surprising. Moreover, the prediction
accuracy and potentially also the created model are completely meaningless.

Thanks For The Fish
-------------------

To implement an ANOVA-based feature selection *properly* we have to do it on
the training dataset **only**. The PyMVPA way of doing this is via a
:class:`~mvpa.clfs.meta.FeatureSelectionClassifier`:

>>> fclf = FeatureSelectionClassifier(clf, fsel)

This is a :term:`meta-classifier` and it just needs two things: A basic
classifier to do the actual classification work and a feature selection
object. We can simple re-use the object instances we already had. Now we
got a meta-classifier that can be used just as any other classifier. Most
importantly we can plug it into a cross-validation procedure (almost
identical to the one we had in the beginning).

>>> cvte = CrossValidation(fclf, NFoldPartitioner(),
...                        enable_ca=['stats'])
>>> results = cvte(bin_demo)
>>> print np.round(cvte.ca.stats.stats['ACC%'], 1)
70.8

This is a lot worse and a lot closer to the truth -- or a so-called
:term:`unbiased estimate` of the generalizability of the classifier model.
We can now also run this improved procedure on our original 8-category
dataset.

>>> results = cvte(ds)
>>> print np.round(cvte.ca.confusion.stats['ACC%'], 1)
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
>>> cvte = CrossValidation(fclf, NFoldPartitioner(),
...                        enable_ca=['stats'])
>>> results = cvte(ds)
>>> print np.round(cvte.ca.stats.stats['ACC%'], 1)
70.8

A drop of 8% in accuracy on about 4 times the number of features. This time
we asked for the top 5% of F-scores.

But how do we get the weight, finally? In PyMVPA (almost) each classifier
is accompanied with a so-called :term:`sensitivity analyzer`. This is an
object that knows how to get them from a particular classifier type (since
each classification algorithm hides them in different places). To create
this *analyzer* we can simply ask the classifier to do it:

>>> sensana = fclf.get_sensitivity_analyzer()
>>> type(sensana)
<class 'mvpa.measures.base.FeatureSelectionClassifierSensitivityAnalyzer'>

As you can see, this even works for our meta-classifier. And again this
analyzer is a :term:`processing object` that returns the desired sensitivity
when called with a dataset.

>>> # alt: `sens = load_tutorial_results('res_haxby2001_sens_5pANOVA')`
>>> sens = sensana(ds)
>>> type(sens)
<class 'mvpa.datasets.base.Dataset'>
>>> print sens.shape
(28, 39912)

.. h5save('results/res_haxby2001_sens_5pANOVA.hdf5', sens)

Why do we get 28 sensitivity maps from the classifier? The support vector
machine is an algorithm for binary classification problems. To be able to deal
with this 8-category dataset, the data is internally split into all
possible binary problems (there are exactly 28 of them). The sensitivities
are extracted for all these partial problems.

.. exercise::

  Figure out which sensitivity map belongs to which combination of
  categories.

If you are not interested in this level of detail, we can combine the maps
into one, as we have done with dataset samples before. A feasable
algorithm might be to take the per feature maximum of absolute
sensitivities in any or the maps. The resulting map will be an indication
of the importance of feature for *any* partial classification.

>>> sens_comb = sens.get_mapped(maxofabs_sample())

.. exercise::

  Project this sensitivity map back into the fMRI volume and compare it to
  the searchlight maps of different radii from the :ref:`previous tutorial
  part <chap_tutorial_searchlight>`.

.. map2nifti(ds, sens_comb).save('results/res_haxby2001_sens_maxofabs_5pANOVA.nii.gz')

You might have noticed some imperfection in our recent approach to compute
a full-brain sensitivity map. We derived it from the full dataset, and not
from cross-validation splits of the data. Rectifying it is easy with a
meta-measure. A meta-measure is analogous to a meta-classifier: a measure
that takes a basic measure, adds a processing step to it and behaves like a
measure itself. The meta-measure we want to use is
:class:`~mvpa.measures.base.SplitFeaturewiseMeasure`.

>>> # alt: `sens = load_tutorial_results('res_haxby2001_splitsens_5pANOVA')`
>>> sensana = fclf.get_sensitivity_analyzer(postproc=maxofabs_sample())
>>> cv_sensana = SplitFeaturewiseMeasure(NFoldSplitter(), sensana)
>>> sens = cv_sensana(ds)
>>> print sens.shape
(12, 39912)

.. h5save('results/res_haxby2001_splitsens_5pANOVA.hdf5', sens)

We re-create our basic sensitivity analyzer, this time automatically
applying the post-processing step that combines the sensitivity maps for
all partial classifications. Finally, we plug it into the meta-measure that
uses an :class:`~mvpa.datasets.splitters.NFoldSplitter` to split the
dataset. Afterwards, we can run the analyzer and we get another dataset,
this time with a sensitivity map per each cross-validation split.

We could combine these maps in a similar way as before, but let's look at
the stability of the ANOVA feature selection instead.

>>> ov = MapOverlap()
>>> overlap_fraction = ov(sens.samples > 0)

With the :class:`~mvpa.misc.support.MapOverlap` helper we can easily
compute the fraction of features that have non-zero sensitivities in all
dataset splits.

.. exercise::

  Inspect the ``ov`` object. Acces that statistics map with the fraction
  of per-feature selections across all splits and project them back into
  the fMRI volume to investigate them.

This could be the end of the data processing. However, by using the meta
measure to compute the sensitivity maps we have lost a convenient way to
access the total performance of the underlying classifier. To again gain
access to it, and get the sensitivities at the same time, we can twist the
processing pipeline a bit.

>>> sclf = SplitClassifier(fclf, NFoldSplitter(), enable_ca=['confusion'])
>>> cv_sensana = sclf.get_sensitivity_analyzer()
>>> sens = cv_sensana(ds)
>>> print sens.shape
(336, 39912)
>>> print cv_sensana.clf.ca.confusion.matrix
[[ 5  0  3  0  0  3  0  1]
 [ 0  9  0  0  0  0  0  0]
 [ 0  2  4  0  0  1  0  0]
 [ 2  1  0 12  0  0  0  0]
 [ 0  0  0  0 12  0  0  0]
 [ 3  0  4  0  0  6  2  1]
 [ 0  0  1  0  0  0 10  0]
 [ 2  0  0  0  0  2  0 10]]

I guess that deserves some explanation. We wrap our
:class:`~mvpa.clfs.meta.FeatureSelectionClassifier` with a new thing, a
:class:`~mvpa.clfs.meta.SplitClassifier`. This is another meta classifier
that performs splitting of a dataset and runs training (and prediction) on
each of the dataset splits separately. It can effectively perform a
cross-validation analysis internally, and we ask it to compute a confusion
matrix of it. The next step is to get a sensitivity analyzer for this meta
meta classifier (this time no post-processing). Once we have got that, we
can run the analysis, we get all sensitivity maps from all internally
trained classifiers. Moreover, the meta sensitivity analyzer also allows
access to its internal meta meta classifier that provides us with the
confusion statistics. Yeah!

While we are at it, it is worth mentioning that the scenario above can be
further extendend. We could add more selection or pre-processing steps
into the classifier, like projecting the data onto PCA components and
limit the classifier to the first 10 components -- for each split. PyMVPA
offers even more complex meta classifiers (e.g.
:class:`~mvpa.clfs.meta.TreeClassifier`) that might be very helpful in some
analysis scenarios.


Closing Words
-------------

We have seen that sensitivity analyses are a useful approach to localize
information that is less contraint and less demanding than a searchlight
analysis.  Specifically, we can use it to discover signals that are
distributed throughout the whole set of features (e.g. the full brain),
but we could also perform an ROI-based analysis with it. It is less
computantionally demanding as we only train the classifier on one set of
features and not thousands, which results in a significant reduction of
required CPU time.

However, there are also caveats. While sensitivities are a much more
direct measure of feature importance as prediction accuracies are, being
close to the bare metal of classifiers also has problems. Depending on the
actual classification algorithm sensitivities might mean something
completely different when compared across classififers. For example, the
popular SVM algorithm solves the classification problem by identifying the
data samples that are *most tricky* to model. The extracted sensitivities
reflect this property. Other algorithms, such as "Gaussian Naive Bayes"
(:class:`~mvpa.clfs.gnb.GNB`) make assumptions about the distribution of
the samples in each category. GNB sensitivities *might* look completely
different, even if both classifiers perform at comparable accuracy levels.
Note, however, that these properties can also be used to address related
research questions.

It should also be noted that sensitivities can not be directly compared to
each other, even if they stem from the same algorithm and are just
computed on different dataset splits. In an analysis one would have to
normalize them first. PyMVPA offers, for example,
:func:`~mvpa.misc.transformers.l1_normed` and
:func:`~mvpa.misc.transformers.l2_normed` that can be used in conjunction
with :class:`~mvpa.mappers.fx.FxMapper` to do that as a post-processing
step.

In this tutorial part we also touched the surface of another important
topic: :term:`feature selection`. We performed an ANOVA-based feature
selection prior to classification to help SVM achieve acceptable
performance. One might wonder if that was a clever idea, since a
*univariate* feature selection step prior to a *multivariate* analysis
somewhat contradicts the goal to identify *multivariate* signals. Only
features will be retained that show some signal on their own. If that
turns out to be a problem for a particular analysis, PyMVPA offers a
number of multivariate alternatives for features selection. There is an
implementation of :term:`recursive feature selection`
(:class:`~mvpa.featsel.rfe.RFE`), and also all classifier sensitivities
can be used to select features. For classifiers where sensitivities cannot
easibly be extracted PyMVPA provides a noise perturbation measure
(:class:`~mvpa.measures.noiseperturbation.NoisePerturbationSensitivity`;
see :ref:`Hanson et al. (2004) <HMH04>` for an example application).

With these building blocks it is possible to run fairly complex analyses.
However, interpreting the results might not always be straight-forward. In
the :ref:`next tutorial part <chap_tutorial_eventrelated>` we will set out
to take away another constraint of all our previously performed analyses. We
are going to go beyond spatial analyses and explore the time dimension.

.. only:: html

  References
  ==========

  .. autosummary::
     :toctree: generated

     ~mvpa.measures.base.Sensitivity
     ~mvpa.featsel.base.SensitivityBasedFeatureSelection
     ~mvpa.measures.base.SplitFeaturewiseMeasure
     ~mvpa.clfs.meta.FeatureSelectionClassifier
     ~mvpa.clfs.meta.SplitClassifier
     ~mvpa.clfs.meta.TreeClassifier
     ~mvpa.clfs.gnb.GNB
     ~mvpa.datasets.splitters.NFoldSplitter
     ~mvpa.misc.support.MapOverlap
     ~mvpa.mappers.fx.FxMapper
     ~mvpa.misc.transformers.l1_normed
     ~mvpa.misc.transformers.l2_normed
     ~mvpa.featsel.rfe.RFE
     ~mvpa.measures.noiseperturbation.NoisePerturbationSensitivity
