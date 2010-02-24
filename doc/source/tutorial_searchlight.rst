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

******************
Part 5: Searchlite
******************

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
Both are objects that get instantiated (potentially with some custom
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
During instantiation an arbitray mapper can be specified that is called
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

>>> sl = sphere_searchlight(cvte, radius=3, postproc=mean_sample())

This single line configures a searchlight analysis that runs a full
cross-validation in every possible sphere in the dataset. Each sphere has a
radius of three voxels. The algorithm uses the coordinates (by default
``voxel_indices``) stored in a feature attribute of the input dataset to
determine local neighborhoods. From the ``postproc`` argument you might
have guessed that this object is also a measure -- and your are right. This
measure returns whatever value is computed by the basic measure (here this
is a cross-validation) and assignes it to the feature representing the
center of the sphere in the output dataset. For this initial example we are
not interested in the full cross-validation output (error per each fold),
but only in the mean error, hence we are using an appropriate mapper for
post-processing. As with any other :term:`processing object` we have to
call it with a dataset to run the actual analysis:

>>> res = sl(ds)
>>> print res
<Dataset: 1x577@float64, <sa: cv_fold>, <a: mapper>>

That was it. However, this was just a toy example with only our ventral
temporal ROI. Let's now run it on a much larger volume, so we can actually
localize something (even loading and preprocessing will take a few seconds).
We will reuse the same searchlight setup and run it on this data as well.
Due to the size of the data it might take a few minutes to compute the
results, depending on the number of CPU in the system.

>>> # instead if actually running this code you can also load the results by
>>> # `res = load_tutorial_results('sl_roi0_results')`
>>> ds = get_haxby2001_data_alternative(roi=0)
>>> print ds.nfeatures
34888
>>> res = sl(ds)

Now let's see what we got. Since a vector with 35k elements is a little
hard to comprehend we have to resort to some statistics.

>>> sphere_errors = res.samples[0]
>>> res_mean = N.mean(res)
>>> res_std = N.std(res)
>>> # we deal with errors here, hence 1.0 minus
>>> chance_level = 1.0 - (1.0 / len(ds.uniquetargets))
>>> print chance_level, N.round(res_mean, 3), N.round(res_std, 3)
0.875 0.848 0.094

Well, the mean empirical error is just barely below the chance level.
However, we would not expect a signal for perfect classification
performance in all spheres anyway. Let's see for how many spheres the error
is more the two standard deviations lower than chance.

>>> print N.round(N.mean(sphere_errors < chance_level - 2 * res_std), 3)
0.091

So in almost 10% of all spheres the error is subtantially lower than what
we would expect for random guessing of the classifier -- that is more than
3000 spheres!

.. exercise::

  Look at the distribution of the errors
  (hint: ``hist(sphere_errors, bins=N.linspace(0, 1, 18))``.
  What do you think in how many spheres the classifier actually picked up
  real signal? What would be a good value to threshold the errors to
  distinguish false from true positives? Think of it in the context of
  statistical testing of fMRI data results. What problems are we facing
  here?

  Once you are done thinking about that -- and only *after* you're done,
  project the sphere error map back into the fMRI volume and look at it as
  a brain overlay in your favorite viewer (hint: you might want to store
  accuracies instead of errors, if your viewer cannot visualize the lower
  tail of the distribution:
  ``map2nifti(ds, 1.0 - sphere_errors).save('sl.nii.gz')``).
  Did looking at the image change your mind?

..
 # figure for the error distribution (emprical and binomial)
 bins = 18
 distr = []
 for i in xrange(100):
     # random binomial variable with errors for each sphere
     r= 1.0 - (stats.binom.rvs(len(ds),
                               1.0 / len(ds.uniquetargets),
                               size=ds.nfeatures) / float(len(ds)))
     distr.append(histogram(r, range=(0, 1), bins=bins, normed=True)[0])
 distr = N.array(distr)
 loc = hist(sphere_errors, range=(0, 1), bins=bins, normed=True)[1]
 plot(loc[:-1] + 1.0/bins/2, distr.mean(axis=0), 'rx--')
 ylim(0,6)
 axvline(0.875, color='red', linestyle='--')
 axvline(res_mean, color='0.3', linestyle='--')

For real!
---------

Now that we have an idea of what can happen in a searchlight analysis,
let's do another one, but this time on a more familiar ROI -- the full brain.

.. exercise::

  Load the dataset with ``get_haxby2001_data_alternative(roi='brain')``
  this will apply any required preprocessing for you. Now run a searchlight
  analysis for radii 0, 1 and 3. For each resulting error map look at the
  distribution of values, project them back into the fMRI volume and
  compare them. How does the distribution change with radius and how does
  it compare to results of the previous exercise? What would be a good
  choice for the threshold in this case?


You have now performed a number of searchlight analyses, investigated the
results and probably tried to interpret them. What conclusions did you draw
from these analyses in terms of the neuroscientific aspects. What have you
learned about object representation in the brain? In this case we have run
8-way classification analyses and we have looked at the average error rate
of thousands of sphere-shaped ROIs in the brain. In some spheres the
classifier could perform perfect classification, i.e. it could predict all
samples equally well. However, this only applies to a handful of over 30k
spheres we have tested. For the vast majority we observe errors somewhere
between the theoretical chance level and zero and we don't know what caused
the error to decrease. We don't even know which samples get misclassified.

From the :ref:`previous tutorial part <chap_tutorial_classifiers>` we know
that there is a way out of this dilemma. We can look at the confusion
matrix of a classifier to get a lot more information that is otherwise
hidden. However, we cannot reasonably do this for thousands of searchlight
spheres. It becomes obvious that a searchlight analysis is probably not the
end of a data exploration, as it raises more questions than it answers.

Moreover, a searchlight cannot detect signals that extend beyond a small
local neighborhood. This property effectively limits the scope of analyses
that can employ this strategy. A study looking a global brain circuitry
will hardly restrict the analysis to patches of few cubic millimeters of
brain tissue. Searchlights also have another nasty aspect. Although they provide
us with a multivariate localization measure, they also inherit the curse of
univariate fMRI data analysis -- `multiple comparisons`_. The :ref:`next
tutorial part <chap_tutorial_sensitivity>` will offers some alternatives
that are more gentle in this respect.

.. _multiple comparisons: http://en.wikipedia.org/wiki/Multiple_comparisons

Despite these limitations a searchlight analysis can be a valuable
exporative tool if used appropriately. The capabilities of PyMVPA's searchlight
implementation go beyond what we looked at in this tutorial. It is not only
possible to run *spatial* searchlights, but multiple spaces can be
considered simultaneously. We will get back to these more advanced topics later
on.



.. only:: html

  References
  ==========

  .. autosummary::
     :toctree: generated

     ~mvpa.measures.searchlight.Searchlight
     ~mvpa.measures.searchlight.sphere_searchlight
