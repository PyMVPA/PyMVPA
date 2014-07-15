.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. vi: set ft=rst sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. index:: Tutorial
.. _chap_tutorial_searchlight:

**************************************
Looking here and there -- Searchlights
**************************************

.. note::

  This tutorial part is also available for download as an `IPython notebook
  <http://ipython.org/ipython-doc/dev/interactive/htmlnotebook.html>`_:
  [`ipynb <notebooks/tutorial_searchlight.ipynb>`_]

In :ref:`chap_tutorial_classifiers` we have seen how we can implement a
classification analysis, but we still have no clue about where in the brain (or
our chosen ROIs) our signal of interest is located.  And that is despite the
fact that we have analyzed the data repeatedly, with different classifiers and
investigated error rates and confusion matrices. So what can we do?

Ideally, we would like to have some way to estimate a score for each feature
that indicates how important that particular feature (most of the time a
voxel) is in the context of a certain classification task. There are various
possibilities to get a vector of such per-feature scores in PyMVPA. We could
simply compute an ANOVA_ F-score per each feature, yielding scores that would
tell us which features vary significantly between any of the categories in our
dataset.

.. _ANOVA: http://en.wikipedia.org/wiki/Analysis_of_variance

Before we can take a look at the implementation details, let's first recreate
our preprocessed demo dataset. The code is very similar to that from
:ref:`chap_tutorial_classifiers` and should raise no questions. We get a
dataset with one sample per category per run.

>>> from mvpa2.tutorial_suite import *
>>> # alt: `ds = load_tutorial_results('ds_haxby2001')`
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

If the code snippet above is of no surprise then you probably got the basic
idea. We created an object instance ``aov`` being a
:class:`~mvpa2.measures.anova.OneWayAnova`. This instance is subsequently
*called* with a dataset and yields the F-scores wrapped into a
:class:`~mvpa2.datasets.base.Dataset`. Where have we seen this before?  Right!
This one differs little from a call to
:class:`~mvpa2.measures.base.CrossValidation`.  Both are objects that get
instantiated (potentially with some custom arguments) and yield the results in
a dataset when called with an input dataset. This is called a :term:`processing
object` and is a common concept in PyMVPA.

However, there is a difference between the two processing objects.
:class:`~mvpa2.measures.base.CrossValidation` returns a dataset with a single
feature -- the accuracy or error rate, while
:class:`~mvpa2.measures.anova.OneWayAnova` returns a vector with one value per
feature. The latter is called a
:class:`~mvpa2.measures.base.FeaturewiseMeasure`. But other than the number of
features in the returned dataset there is not much of a difference. All
measures in PyMVPA, for example, support an optional post-processing step.
During instantiation of a measure an arbitrary mapper can be specified to be
called internally to forward-map the results before they are returned. If, for
some reason, the F-scores need to be scaled into the interval [0,1], an
:class:`~mvpa2.mappers.fx.FxMapper` can be used to achieve that:

>>> aov = OneWayAnova(
...         postproc=FxMapper('features',
...                           lambda x: x / x.max(),
...                           attrfx=None))
>>> f = aov(ds)
>>> print f.samples.max()
1.0

.. map2nifti(ds, f).to_filename('results/res_haxby2001_fscore_vt.nii.gz')

.. exercise::

  Map the F-scores back into a brain volume and look at their distribution
  in the ventral temporal ROI.

Now that we know how to compute feature-wise F-scores we can start worrying
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

We know almost all pieces to implement a searchlight analysis in
PyMVPA. We can load and preprocess datasets, we can set up a
cross-validation procedure.

>>> clf = kNN(k=1, dfx=one_minus_correlation, voting='majority')
>>> cvte = CrossValidation(clf, HalfPartitioner())

The only thing left is that we have to split the dataset into all possible
sphere neighborhoods that intersect with the brain. To achieve this, we
can use :func:`~mvpa2.measures.searchlight.sphere_searchlight`:

>>> sl = sphere_searchlight(cvte, radius=3, postproc=mean_sample())

This single line configures a searchlight analysis that runs a full
cross-validation in every possible sphere in the dataset. Each sphere has a
radius of three voxels. The algorithm uses the coordinates (by default
``voxel_indices``) stored in a feature attribute of the input dataset to
determine local neighborhoods. From the ``postproc`` argument you might
have guessed that this object is also a measure -- and your are right. This
measure returns whatever value is computed by the basic measure (here this
is a cross-validation) and assigns it to the feature representing the
center of the sphere in the output dataset. For this initial example we are
not interested in the full cross-validation output (error per each fold),
but only in the mean error, hence we are using an appropriate mapper for
post-processing. As with any other :term:`processing object` we have to
call it with a dataset to run the actual analysis:

>>> res = sl(ds)
>>> print res
<Dataset: 1x577@float64, <sa: cvfolds>, <fa: center_ids>, <a: mapper>>

That was it. However, this was just a toy example with only our ventral
temporal ROI. Let's now run it on a much larger volume, so we can actually
localize something (even loading and preprocessing will take a few seconds).
We will reuse the same searchlight setup and run it on this data as well.
Due to the size of the data it might take a few minutes to compute the
results, depending on the number of CPUs in the system.

>>> # alt: `ds = load_tutorial_results('ds_haxby2001_alt_roi0')`
>>> ds = get_haxby2001_data_alternative(roi=0)
>>> print ds.nfeatures
34888
>>> # alt: `res = load_tutorial_results('res_haxby2001_sl_avgacc_roi0')`
>>> res = sl(ds)

.. h5save("results/ds_haxby2001_alt_roi0.hdf5", ds, compression=9)
.. h5save('results/res_haxby2001_sl_avgacc_roi0.hdf5', res)

Now let's see what we got. Since a vector with 35k elements is a little
hard to comprehend we have to resort to some statistics.

>>> sphere_errors = res.samples[0]
>>> res_mean = np.mean(res)
>>> res_std = np.std(res)
>>> # we deal with errors here, hence 1.0 minus
>>> chance_level = 1.0 - (1.0 / len(ds.uniquetargets))

.. map2nifti(ds, 1.0 - sphere_errors).to_filename('results/res_haxby2001_sl_avgacc_roi0.nii.gz')

As you'll see, the mean empirical error is just barely below the chance level.
However, we would not expect a signal for perfect classification
performance in all spheres anyway. Let's see for how many spheres the error
is more the two standard deviations lower than chance.

>>> frac_lower = np.round(np.mean(sphere_errors < chance_level - 2 * res_std), 3)

So in almost 10% of all spheres the error is substantially lower than what
we would expect for random guessing of the classifier -- that is more than
3000 spheres!

.. exercise::

  Look at the distribution of the errors
  (hint: ``hist(sphere_errors, bins=np.linspace(0, 1, 18))``.
  In how many spheres do you think the classifier actually picked up
  real signal? What would be a good value to threshold the errors to
  distinguish false from true positives? Think of it in the context of
  statistical testing of fMRI data results. What problems are we facing
  here?

  Once you are done thinking about that -- and only *after* you're done,
  project the sphere error map back into the fMRI volume and look at it as
  a brain overlay in your favorite viewer (hint: you might want to store
  accuracies instead of errors, if your viewer cannot visualize the lower
  tail of the distribution:
  ``map2nifti(ds, 1.0 - sphere_errors).to_filename('sl.nii.gz')``).
  Did looking at the image change your mind?

..
 # figure for the error distribution (empirical and binomial)
 bins = 18
 distr = []
 for i in xrange(100):
     # random binomial variable with errors for each sphere
     r= 1.0 - (stats.binom.rvs(len(ds),
                               1.0 / len(ds.uniquetargets),
                               size=ds.nfeatures) / float(len(ds)))
     distr.append(histogram(r, range=(0, 1), bins=bins, normed=True)[0])
 distr = np.array(distr)
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

.. h5save('results/ds_haxby2001_alt_brain.hdf5', ds)
.. h5save('results/res_haxby2001_sl_avgacc_r0_brain.hdf5', r0)
.. map2nifti(ds, 1.0 - r0.samples[0]).to_filename('results/res_haxby2001_sl_avgacc_r0_brain.nii.gz')
.. h5save('results/res_haxby2001_sl_avgacc_r1_brain.hdf5', r1)
.. map2nifti(ds, 1.0 - r1.samples[0]).to_filename('results/res_haxby2001_sl_avgacc_r1_brain.nii.gz')
.. h5save('results/res_haxby2001_sl_avgacc_r3_brain.hdf5', r3)
.. map2nifti(ds, 1.0 - r3.samples[0]).to_filename('results/res_haxby2001_sl_avgacc_r3_brain.nii.gz')

You have now performed a number of searchlight analyses, investigated the
results and probably tried to interpret them. What conclusions did you draw
from these analyses in terms of the neuroscientific aspects? What have you
learned about object representation in the brain? In this case we have run
8-way classification analyses and have looked at the average error rate across
all conditions in thousands of sphere-shaped ROIs in the brain. In some spheres the
classifier could perform well, i.e. it could predict all
samples equally well. However, this only applies to a handful of over 30k
spheres we have tested, and does not reveal whether the classifier was capable of
classifying *all* of the conditions or just some.  For the vast majority
we observe errors somewhere
between the theoretical chance level and zero and we don't know what caused
the error to decrease. We don't even know which samples get misclassified.

From :ref:`chap_tutorial_classifiers` we know
that there is a way out of this dilemma. We can look at the confusion
matrix of a classifier to get a lot more information that is otherwise
hidden. However, we cannot reasonably do this for thousands of searchlight
spheres (Note that this is not completely true. See e.g. :ref:`Connolly et al.,
2012 <CGG+12>` for some creative use-cases for searchlights).
It becomes obvious that a searchlight analysis is probably not the
end of a data exploration but rather a crude take off,
as it raises more questions than it answers.

Moreover, a searchlight cannot detect signals that extend beyond a small
local neighborhood. This property effectively limits the scope of analyses
that can employ this strategy. A study looking a global brain circuitry
will hardly restrict the analysis to patches of a few cubic millimeters of
brain tissue. As we have seen before, searchlights also have another nasty
aspect. Although they provide us with a multivariate localization measure,
they also inherit the curse of univariate fMRI data analysis --
`multiple comparisons`_.

.. _multiple comparisons: http://en.wikipedia.org/wiki/Multiple_comparisons

Despite these limitations a searchlight analysis can be a valuable
exploratory tool if used appropriately. The capabilities of PyMVPA's searchlight
implementation go beyond what we looked at in this tutorial. It is not only
possible to run *spatial* searchlights, but multiple spaces can be
considered simultaneously.
