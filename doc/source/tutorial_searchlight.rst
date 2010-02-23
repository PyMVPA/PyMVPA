.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. index:: Tutorial
.. _chap_tutorial_searchlight:

******************************
Part 5: The Mighty Searchlight
******************************

The :ref:`previous tutorial part <chap_tutorial_classifiers>` ended with
the insight that we still have no clue about where in the brain (or our
chosen ROIs) the signal is located that is picked up by the classifier.
And that is despite the fact that we have analyzed the data repeatedly,
with different classifiers and investigated error rates and confusion
matrices. So what can we do?

Ideally, we would have something that computes a vector with a per feature
score that indicates how important a particular feature (most of the time a
voxel) is in the context of a certain classification task. There are various
possibilities to get such a vector in PyMVPA. We could simply compute an
ANOVA_ F-score per feature, yielding a score that tell us which feature
varies significantly between any of the categories in our dataset. 

.. _ANOVA: http://en.wikipedia.org/wiki/Analysis_of_variance

Before we can take a look at the implementation details, let's first
recreate our preprocessed demo dataset. The code is taken verbatim from the 
:ref:`previous tutorial part <chap_tutorial_classifiers>` and should raise
no questions. We get a dataset with one sample per category per run.

>>> from tutorial_lib import *
>>> ds = get_haxby2001_data(roi='vt')
>>> ds.shape
(16, 577)


Measures
--------

Now that we have the dataset, computing the desired ANOVA F-scores is
relatively painless:

>>> aov = OneWayAnova()
>>> f = aov(ds)
>>> print f
<Dataset: 1x577@float64, <fa: fprob>>

If the code snippet above is no surprise then you probably got the basic
idea. We created an object instance ``aov`` being a
:class:`~mvpa.measures.anova.OneWayAnova`. This instance is subsequently
*called* with a dataset and yields the F-scores -- in a
:class:`~mvpa.datasets.base.Dataset`. Where have we seen this before?
Right! That is little different from a call to
:class:`~mvpa.algorithms.cvtranserror.CrossValidatedTransferError`.
Both are objects that get instanciated (potentially with some custom
arguments) and yield the results in a dataset when called with an input
dataset. This is called a :term:`processing object` and is a common
concept in PyMVPA.

However, there is a difference between the two processing objects.
:class:`~mvpa.algorithms.cvtranserror.CrossValidatedTransferError` returns
a dataset with a single feature -- the accuracy or error rate, while
:class:`~mvpa.measures.anova.OneWayAnova` returns a vector with one value
per feature. The latter is called a
:class:`~mvpa.measures.base.FeaturewiseDatasetMeasure`. But other than the
number of features in the returned dataset there is little difference. All
measures in PyMVPA, for example, support an optional post-processing step.
During instanciation an arbitray mapper can be specified that is called
internally to forward-map the results before they are returned. If, for
some reason, the F-scores need to be scaled into the interval [0,1], an
:class:`~mvpa.mappers.fx.FxMapper` can be used to achieve that:

>>> aov = OneWayAnova(
...         postproc=FxMapper('samples',
...                           lambda x: x / x.max()))
>>> f = aov(ds)
>>> print f.samples.max()
1.0

.. exercise::

  Map the F-scores back into a brain volume and look at their distribution
  in the ventral temporal ROI.

Now that we know how to compute featurewise F-scores we can start worrying
about them. Our original goal was to decipher information that is encoded
in the multivariate pattern of brain activation. But now we are using an
ANOVA, a **univariate** measure, to localize important voxels? There must
be something else -- and there is!


Searching, searching, searching, ...
------------------------------------

:ref:`Kriegeskorte et al. (2006) <KGB06>` suggested an algorithm that takes
a small, sphere-shaped neighborhood of brain voxels and computes a
multivariate measure to quantify the amount of information encoded in its
pattern (e.g.  `mutual information`_). Later on this :term:`searchlight`
approach has been extended to run a full classifier cross-validation in
every possible sphere in the brain. Since that, multiple studies have
employed this approach to localize relevant information in a locally
constraint fashion.

.. _mutual information: http://en.wikipedia.org/wiki/Mutual_information

We almost know all the pieces to implement a searchlight analyses in
PyMVPA. We can load and preprocess datasets, we can set up a
cross-validation procedure.

>>> clf = kNN(k=1, dfx=one_minus_correlation, voting='majority')
>>> terr = TransferError(clf)
>>> cvte = CrossValidatedTransferError(terr, splitter=HalfSplitter())

The only thing left is that we have to split the dataset into all possible
sphere neighborhoods that intersect with the brain. To achieve this, we
can use :func:`~mvpa.measures.searchlight.sphere_searchlight`:

>>> sl = sphere_searchlight(cvte, postproc=mean_sample())

This single line configures a searchlight analysis that runs a full
cross-validation in every possible sphere in the dataset. The algorithm
uses the coordinates (by default ``voxel_indices``) stored in a feature
attribute of the input dataset to determine local neighborhoods. From the
``postproc`` argument you might have guessed that this object is also a
measure -- and your are right. This measure returns whatever value is
computed by the basic measure (here this is a cross-validation) and
assignes it to the feature representing the center of the sphere in the
output dataset. For this initial example we are not interested in the full
cross-validation output (error per each fold), but only in the mean error,
hence we are using an appropriate mapper for post-processing. As with any
other :term:`processing object` we have to call it with a dataset to run
the actual analysis:

#$>>> res = sl(ds)
>>> print res
<Dataset: 1x577@float64, <sa: cv_fold>, <a: mapper>>

That was it. However, this was just a toy example with only our ventral
temporal ROI. Let's now run it on a much larger volume, so we can actually
localize something (even loading and preprocessing will take a few seconds).
We will reuse the same searchlight setup and run it on this data as well.
Due to the size of the data it might take a few minutes to compute the
results, depending on the number of CPU in the system.

>>> ds = get_haxby2001_data_alternative(roi=0)
>>> print ds.nfeatures
34888
>>> res = sl(ds)

>>> h = hist(res.samples[0], bins=len(N.unique(res.samples[0]))+1)


ds = get_haxby2001_data_alternative(roi=0)
clf = kNN(k=1, dfx=one_minus_correlation, voting='majority')
terr = TransferError(clf)
cvte = CrossValidatedTransferError(terr, splitter=HalfSplitter(attr='runtype'))
sl = sphere_searchlight(cvte, postproc=mean_sample())
res=sl(ds)
map2nifti(ds, 1 - res.samples).save('sl.nii.gz')


.. only:: html

  References
  ==========

  .. autosummary::
     :toctree: generated

     ~mvpa.measures.searchlight.Searchlight
