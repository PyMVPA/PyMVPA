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
>>> # directory that contains the data files
>>> datapath = os.path.join(pymvpa_datadbroot,
...                         'demo_blockfmri', 'demo_blockfmri')
>>> # load the raw data
>>> attr = SampleAttributes(os.path.join(datapath, 'attributes.txt'))
>>> ds = fmri_dataset(samples=os.path.join(datapath, 'bold.nii.gz'),
...                   targets=attr.targets, chunks=attr.chunks,
...                   mask=os.path.join(datapath, 'mask_vt.nii.gz'))
>>> # pre-process
>>> poly_detrend(ds, polyord=1, chunks='chunks')
>>> zscore(ds, param_est=('targets', ['rest']))
>>> ds = ds[ds.sa.targets != 'rest']
>>> # average
>>> run_averager = mean_group_sample(['targets', 'chunks'])
>>> ds = ds.get_mapped(run_averager)
>>> ds.shape
(96, 577)


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


Searching, searching, searching
-------------------------------

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
