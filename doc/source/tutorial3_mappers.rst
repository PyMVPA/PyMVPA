.. -*- mode: rst; fill-column: 78 -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. index:: Tutorial
.. _chap_tutorial3:

***************************************
Part 3: Mappers -- The Swiss Army Knife
***************************************

In the :ref:`previous tutorial part <chap_tutorial2>` we have discovered a
magic ingredient of datasets: a mapper. Mappers are probably the most
powerful concept in PyMVPA, and there is little one would do without them.
As a matter of fact, even in the :ref:`first tutorial part
<chap_tutorial1>` we have used them already, without even seeing them.

In general, a mapper is an algorithm that transforms data.
This transformation can be as simple as selecting a subset of data, or as
complex as a multi-stage preprocessing pipeline. Some transformations are
reversible, others are not. Some are simple one-step computations, others
are iterative algorithms that have to be trained on data before they can be
used. In PyMVPA, all these transformations are :mod:`~mvpa.mappers`.

Let's create a dummy dataset (5 samples, 12 features). This time we will use a
new method to create the dataset, the `dataset_wizard`. Here it is fully
equivalent to a regular constructor call (i.e.  `~mvpa.datasets.base.Dataset`),
but we will shortly see some nice convenience aspects.

  >>> from mvpa.suite import *
  >>> ds = dataset_wizard(N.ones((5, 12)))
  >>> ds.shape
  (5, 12)

A mapper is a :term:`dataset attribute`, hence it is stored in the
corresponding attribute collection. However, not every dataset actually has
a mapper. For example, the simple one we have just created doesn't have any:

  >>> 'mapper' in ds.a
  False

Now let's look at a very similar dataset that only differs in a tiny but
a very important detail:

  >>> ds = dataset_wizard(N.ones((5, 4, 3)))
  >>> ds.shape
  (5, 12)
  >>> 'mapper' in ds.a
  True
  >>> print ds.a.mapper
  <FlattenMapper>

We see that the resulting dataset looks identical to the one above, but this time
it got created from a 3D samples array (i.e. five samples, where each is a 4x3
matrix). Somehow this 3D array got transformed into a 2D samples array in the
dataset. This magic behavior is unveiled by observing that the dataset's mapper
is a `~mvpa.mappers.flatten.FlattenMapper`.

The purpose of this mapper is precisely what we have just observed: reshaping
data arrays into 2D. It does it by preserving the first axis (in PyMVPA datasets
this is the axis that separates the samples) and concatenates all other axis
into the second one.

A very important feature of this mapper is that this transformation is
reversible. We can simply ask the mapper to put our samples back into the
original 3D shape.

  >>> orig_data = ds.a.mapper.reverse(ds.samples)
  >>> orig_data.shape
  (5, 4, 3)

In interactive scripting sessions this is would be a relatively bulky command to
type, although it might be quite frequently used. To make ones fingers suffer
less there is a little shortcut that does exactly the same:

  >>> orig_data = ds.O
  >>> orig_data.shape
  (5, 4, 3)

Since mappers represent particular transformations they can also be seen as a
protocol of what has been done. If we look at the dataset, we nkow that it had
been flattened on the way from its origin to a samples array in a dataset. This
feature can become really useful, if the processing become more complex. Let's
look at a possible next step -- selecting a subset of interesting features:

  >>> myfavs = [1, 2, 8, 10]
  >>> subds = ds[:, myfavs]
  >>> subds.shape
  (5, 4)
  >>> 'mapper' in subds.a
  True
  >>> print subds.a.mapper
  <ChainMapper: <Flatten>-<FeatureSlice>>

Now the situation has changed: *two* new mappers appeared in the dataset -- a
`~mvpa.mappers.base.ChainMapper` and a `~mvpa.mappers.base.FeatureSliceMapper`.
The latter described (and actually performs) the slicing operation we just made,
while the former encapsulates the two mappers into a processing pipeline.
We can see that the mapper chain represents the processing history of the
dataset like a breadcrumb track.

It is important to realize that the `~mvpa.mappers.base.ChainMapper` is a fully
features mapper that can also be used as such.

  >>> ds.O.shape
  (5, 4, 3)

As it has been mentioned, mappers can not only transform a single dataset, but
can be feed with other data (as long as it is compatible with the mapper. Let's
look at a reverse-mapping of the chain first.

  >>> subds.nfeatures
  4
  >>> revtest = N.arange(subds.nfeatures) + 10
  >>> print revtest
  [10 11 12 13]
  >>> rmapped = subds.a.mapper.reverse1(revtest)
  >>> rmapped.shape
  (4, 3)
  >>> print rmapped
  [[ 0 10 11]
   [ 0  0  0]
   [ 0  0 12]
   [ 0 13  0]]

Reverse mapping of a single sample (one-dimensional feature vector) through the
mapper chain created a 4x3 array that corresponds to the dimensions of a sample
in our original data space. Moreover, we see that each feature value is
precisely placed into the position that corresponds to the features selected
in the previous dataset slicing operation. And now for the forward mapping:

  >>> fwdtest = N.arange(12).reshape(4,3)
  >>> print fwdtest
  [[ 0  1  2]
   [ 3  4  5]
   [ 6  7  8]
   [ 9 10 11]]
  >>> fmapped = subds.a.mapper.forward1(fwdtest)
  >>> fmapped.shape
  (4,)
  >>> print fmapped
  [ 1  2  8 10]

Although `subds` has less features than our input data, forward mapping applies
the same transformation that had been done to the dataset itself also to our
test 4x3 array. The procedure yields a feature vector of the same shape as the
one in `subds`. By looking at the forward-mapped data, we can verify that the
correct features have been chosen.

Doing ``get_haxby2001_data()`` From Scratch
===========================================

Now we have pretty much all the pieces that we need to perform a full
cross-validation analysis. Remember, in :ref:`part one of the tutorial
<chap_tutorial1>` we cheated a bit, by using a magic function to load the
preprocessed fMRI data. This time we are more prepared. We know how to
load fMRI data from timeseries images, we know how to add and access
attributes in a dataset, we know how to slice datasets, and we know that
we can manipulate datasets with mappers.

Our goal is now to combine all these little pieces into code that produces
the dataset we already used at beginning. That is:

  A *pattern of activation* for each stimulus category in each half of the
  data (split by odd vs. even runs; i.e. 16 samples), including the
  associated :term:`sample attribute`\ s that are necessary to perform a
  cross-validated classification analysis of the data.

DISCOVER THE CODE STEP BY STEP
::

   def get_haxby2001_data(path=os.path.join(pymvpa_dataroot,
                                         'demo_blockfmri',
                                         'demo_blockfmri')):
    attr = SampleAttributes(os.path.join(path, 'attributes.txt'))
    ds = fmri_dataset(samples=os.path.join(path, 'bold.nii.gz'),
                      labels=attr.labels, chunks=attr.chunks,
                      mask=os.path.join(path, 'mask_vt.nii.gz'))

     do chunkswise linear detrending on dataset
    poly_detrend(ds, polyord=1, chunks='chunks', inspace='time_coords')

    # mark the odd and even runs
    rnames = {0: 'even', 1: 'odd'}
    ds.sa['runtype'] = [rnames[c % 2] for c in ds.sa.chunks]

    # compute the mean sample per condition and odd vs. even runs
    # aka "constructive interference"
    ds = ds.get_mapped(mean_group_sample(['labels', 'runtype']))

    # zscore dataset relative to baseline ('rest') mean
    zscore(ds, param_est=('labels', ['rest']))

    # exclude the rest condition from the dataset
    ds = ds[ds.sa.labels != 'rest']

    return ds


.. only:: html

  References
  ==========

  .. autosummary::
     :toctree: generated

     ~mvpa.mappers
     ~mvpa.mappers.base.Mapper
     ~mvpa.mappers.base.FeatureSliceMapper
     ~mvpa.mappers.flatten.FlattenMapper
     ~mvpa.mappers.fx.FxMapper
     ~mvpa.mappers.base.ChainMapper
