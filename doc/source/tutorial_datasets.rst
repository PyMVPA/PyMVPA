.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. vi: set ft=rst sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. index:: Tutorial, Dataset concepts
.. _chap_tutorial_datasets:

***********************************
Part 2: Dataset Basics and Concepts
***********************************

A `~mvpa.datasets.base.Dataset` is the basic data container in PyMVPA. It
serves as the primary form of input data storage, but also as container for
more complex results returned by some algorithm. In this tutorial part we will
take a look at what a dataset consists of, and how it works.

In the simplest case, a dataset only contains *data* that is a matrix of
numerical values.

>>> from tutorial_lib import *
>>> data = [[  1,  1, -1],
...         [  2,  0,  0],
...         [  3,  1,  1],
...         [  4,  0, -1]]
>>> ds = Dataset(data)
>>> ds.shape
(4, 3)
>>> len(ds)
4
>>> ds.nfeatures
3
>>> ds.samples
array([[ 1,  1, -1],
       [ 2,  0,  0],
       [ 3,  1,  1],
       [ 4,  0, -1]])

In the above example, every row vector in the ``data`` matrix becomes an
observation or a :term:`sample` in the dataset, and every column vector
represents an individual variable or a :term:`feature`. The concepts of samples
and features are essential for a dataset, hence we take a further, closer look.

The dataset assumes the first axis of the data to be the samples separating
dimension. If the dataset is created using a one-dimensional vector it will
therefore have as many samples as elements in the vector, and only one feature.

>>> one_d = [ 0, 1, 2, 3 ]
>>> one_ds = Dataset(one_d)
>>> one_ds.shape
(4, 1)

On the other hand, if a dataset is created from multi-dimensional data, only its
second axis represents the features

>>> import numpy as np
>>> m_ds = Dataset(np.random.random((3, 4, 2, 3)))
>>> m_ds.shape
(3, 4, 2, 3)
>>> m_ds.nfeatures
4

In this case we have a dataset with three samples and four features, where each
feature is a 2x3 matrix. In case somebody is wondering now, why not simply each
value in the data array is considered as its own feature (yielding 24 features)
-- stay tuned, as this is going to be of importance later on.


Attributes
==========

What we have seen so far does not really warrant the use of a dataset over a
plain array or a matrix with samples. However, in the MVPA context we often need
to know more about each samples than just the value of its features.  In the
previous tutorial part we have already seen that per-sample :term:`target`
values are required for supervised-learning algorithms, and that a dataset
often has to be split based on the origin of specific groups of samples.  For
this type of auxiliary information a dataset can also contain collections of
three types of :term:`attribute`\ s: :term:`sample attribute`, :term:`feature attribute`, and
:term:`dataset attribute`.

For Samples
-----------

In a dataset each :term:`sample` can have an arbitrary number of additional
attributes. They are stored as vectors of the same length as the number of samples
in a collection, and are accessible via the ``sa`` attribute. A collection is
derived from a standard Python `dict`, and hence adding sample attributes
works identical to adding elements to a dictionary:

>>> ds.sa['some_attr'] = [ 0., 1, 1, 3 ]
>>> ds.sa.keys()
['some_attr']

However, sample attributes are not directly stored as plain data, but for
various reasons as a so-called `~mvpa.base.collections.Collectable` that in
turn embeds a NumPy array with the actual attribute:

>>> type(ds.sa['some_attr'])
<class 'mvpa.base.collections.ArrayCollectable'>
>>> ds.sa['some_attr'].value
array([ 0.,  1.,  1.,  3.])

This "complication" is done to be able to extend attributes with additional
functionality that is often needed and can offer significant speed-up of
processing. For example, sample attributes carry a list of their unique values.
This list is only computed once (upon first request) and can subsequently be
accessed directly without repeated and expensive searches:

>>> ds.sa['some_attr'].unique
array([ 0.,  1.,  3.])

However, for most interactive uses of PyMVPA this type of access to attributes'
``.value`` is relatively cumbersome (too much typing), therefore collections offer direct
attribute access by name:

>>> ds.sa.some_attr
array([ 0.,  1.,  1.,  3.])

Another purpose of the sample attribute collection is to preserve data
integrity, by disallowing improper attributes:

>>> ds.sa['invalid'] = 4
Traceback (most recent call last):
  File "/usr/lib/python2.6/doctest.py", line 1253, in __run
    compileflags, 1) in test.globs
  File "<doctest tutorial_datasets.rst[20]>", line 1, in <module>
    ds.sa['invalid'] = 4
  File "/home/test/pymvpa/mvpa/base/collections.py", line 459, in __setitem__
    value = ArrayCollectable(value)
  File "/home/test/pymvpa/mvpa/base/collections.py", line 171, in __init__
    % self.__class__.__name__)
ValueError: ArrayCollectable only takes sequences as value.

>>> ds.sa['invalid'] = [ 1, 2, 3, 4, 5, 6 ]
Traceback (most recent call last):
  File "/usr/lib/python2.6/doctest.py", line 1253, in __run
    compileflags, 1) in test.globs
  File "<doctest tutorial_datasets.rst[21]>", line 1, in <module>
    ds.sa['invalid'] = [ 1, 2, 3, 4, 5, 6 ]
  File "/home/test/pymvpa/mvpa/base/collections.py", line 468, in __setitem__
    str(self)))
ValueError: Collectable 'invalid' with length [6] does not match the required length [4] of collection '<SampleAttributesCollection: some_attr>'.

But other than basic plausibility checks no further constraints on values of
samples attributes exist. As long as the length of the attribute vector matches
the number of samples in the dataset, and the attributes values can be stored
in a NumPy array, any value is allowed. For example, it is perfectly possible
and supported to store literal attributes. It should also be noted that each
attribute may have its own individual data type, hence it is possible to have
literal and numeric attributes in the same dataset.

>>> ds.sa['literal'] = ['one', 'two', 'three', 'four']
>>> sorted(ds.sa.keys())
['literal', 'some_attr']
>>> for attr in ds.sa:
...    print "%s: %s" % (attr, ds.sa[attr].value.dtype.name)
literal: string40
some_attr: float64



For Features
------------

:term:`Feature attribute`\ s are almost identical to :term:`sample attribute`\
s the *only* difference is that instead of having one attribute value per
sample, feature attributes have one value per (guess what? ...) *feature*.
Moreover, they are stored in a separate collection in the datasets that is
called ``fa``:

>>> ds.nfeatures
3
>>> ds.fa['my_fav'] = [0, 1, 0]
>>> ds.fa['responsible'] = ['me', 'you', 'nobody']
>>> sorted(ds.fa.keys())
['my_fav', 'responsible']


For The Dataset
---------------

Finally, there can be also attributes, not per each sample, or each
feature, but for the dataset as a whole: so called :term:`dataset
attribute`\s. Assigning such attributes and accessing them later on work in
exactly the same way as for the other two types of attributes, except that dataset
attributes are stored in their own collection which is accessible via the
``a`` property of the dataset.  However, in contrast to sample and feature
attribute no constraints on the type or size are imposed -- anything can be
stored. Let's store a list with all files in the current directory, just
because we can:

>>> from glob import glob
>>> ds.a['pointless'] = glob("*")
>>> 'setup.py' in ds.a.pointless
True


Slicing, resampling, feature selection
======================================

At this point we can already construct a dataset from simple arrays and
enrich it with an arbitrary number of additional attributes. But just
having a dataset isn't enough. From part one of this tutorial we already
know that we need to be able to select subsets of a dataset for further
processing, and we also know that this is possible with PyMVPA's datasets.
Now it is time to have a closer look into how it works.

Slicing a dataset (i.e. selecting specific subsets) is very similar to
slicing a NumPy array. It actually works *almost* identical. A dataset
supports Python's `slice` syntax, but also selection by boolean masks, and
indices. The following three slicing operations
result in equivalent output datasets, by always selecting every other samples
in the dataset:

>>> # original
>>> ds.samples
array([[ 1,  1, -1],
       [ 2,  0,  0],
       [ 3,  1,  1],
       [ 4,  0, -1]])
>>>
>>> # Python-style slicing
>>> ds[::2].samples
array([[ 1,  1, -1],
       [ 3,  1,  1]])
>>>
>>> # Boolean mask array
>>> mask = np.array([True, False, True, False])
>>> ds[mask].samples
array([[ 1,  1, -1],
       [ 3,  1,  1]])
>>>
>>> # Slicing by index -- Python indexing start with 0 !!
>>> ds[[0, 2]].samples
array([[ 1,  1, -1],
       [ 3,  1,  1]])

.. exercise::

  Search the `NumPy documentation`_ for the difference between "basic slicing"
  and "advanced indexing". Especially the aspect of memory consumption
  applies to dataset slicing as well, and being aware of this fact might
  help to write more efficient analysis scripts. Which of the three slicing
  approaches above is the most memory-efficient?  Which of the three slicing
  approaches above might lead to unexpected side-effects if output dataset
  gets modified?

.. _NumPy documentation: http://docs.scipy.org/doc/


All three slicing-styles equally applicable to the selection of feature subsets
within a dataset. Remember, features are represented on the second axis
of a dataset.

>>> ds[:, [1,2]].samples
array([[ 1, -1],
       [ 0,  0],
       [ 1,  1],
       [ 0, -1]])

By applying a selection by indices to the second axis, we can easily get
the last two features of our example dataset. Please note the `:` is supplied
as first axis slicing. This is the Python way to indicate *take everything
along this axis*, hence take all samples.

As you can guess, it is also possible to select subsets of samples and
features at the same time.

>>> subds = ds[[0,1], [0,2]]
>>> subds.samples
array([[ 1, -1],
       [ 2,  0]])

If you have prior experience with NumPy you might be confused now. What you
might have expected is this:

>>> ds.samples[[0,1], [0,2]]
array([1, 0])

The above code applies the same slicing directly to the NumPy array with
the samples, and the result is fundamentally different. For NumPy arrays
the style of slicing allows to select specific elements by their indices on
each axis of an array. For PyMVPA's datasets this mode is not very useful,
instead we typically want to select rows and columns, i.e. samples and
features given by their indices.


.. exercise::

  Try to select samples [0,1] and features [0,2,3] simultaneously using
  dataset slicing.  Now apply the same slicing to the samples array itself
  (``ds.samples``) -- make sure that the result doesn't surprise you and find
  a pure NumPy way to achieve similar selection.


One last interesting thing to look at, in the context of dataset slicing
are the attributes. What happens to them when a subset of samples and/or
features is chosen? Our original dataset had both samples and feature attributes:

>>> print ds.sa.some_attr
[ 0.  1.  1.  3.]
>>> print ds.fa.responsible
['me' 'you' 'nobody']

Now let's look at what they became in the subset-dataset we previously
created:

>>> print subds.sa.some_attr
[ 0.  1.]
>>> print subds.fa.responsible
['me' 'nobody']

We see that both attributes are still there and, moreover, also the
appropriate subsets have been selected.


Loading fMRI data
=================

Enough of theoretical foreplay -- let's look at a concrete example of an
fMRI dataset. PyMVPA has several helper functions to load data from
specialized formats, and the one for fMRI data is
`~mvpa.datasets.mri.fmri_dataset()`. The example dataset we are going to
look at is a single subject from Haxby et al. (2001) that we already
loaded in part one of this tutorial. For more convenience, and less typing
we first specify the path of the directory with the fMRI data.

>>> path=os.path.join(tutorial_data_path, 'data')

In the simplest case, we now let `~mvpa.datasets.mri.fmri_dataset` do its job, by just
pointing it to the fMRI data file. The data is stored as a NIfTI file that has
all runs of the experiment concatenated into a single file.

>>> ds = fmri_dataset(os.path.join(path, 'bold.nii.gz'))
>>> len(ds)
1452
>>> ds.nfeatures
163840
>>> ds.shape
(1452, 163840)

We can notice two things. First, it worked! Second, we get a
two-dimensional dataset with 1452 samples (these are volumes in the NIfTI
file), and over 160k features (these are voxels in the volume). The voxels
are represented as a one-dimensional vector, and it seems that they have
lost their association with the 3D-voxel-space. However, this is not the
case, as we will see in the next chapter.  PyMVPA represents
data in this simple format to make it compatible with a vast range of generic
algorithms that expect data to be a simple matrix.

We just loaded all data from that NIfTI file, but usually we would be
interested in a subset only, i.e. "brain voxels".
`~mvpa.datasets.mri.fmri_dataset` is capable of performing data masking. We just need to
specify a mask image. Such mask image is generated in pretty much any fMRI
analysis pipeline -- may it be a full-brain mask computed during
skull-stripping, or an activation map from a functional localizer. We are going
to use the original GLM-based localizer mask of ventral temporal cortex
from Haxby et al. (2001). We already know that it comprises 577 voxels.
Let's reload the dataset:

>>> ds = fmri_dataset(os.path.join(path, 'bold.nii.gz'),
...                   mask=os.path.join(path, 'mask_vt.nii.gz'))
>>> len(ds)
1452
>>> ds.nfeatures
577

As expected, we get the same number of samples and also only 577 features
-- voxels corresponding to non-zero elements in the mask image. Now, let's
explore this dataset a little further.

Besides samples the dataset offers number of attributes that enhance the
data with information that is present in the NIfTI image header in the file. Each sample has
information about its volume id in the time series and the actual acquisition
time (relative to the beginning of the file). Moreover, the original voxel
index (sometimes referred to as ``ijk``) for each feature is available too.
Finally, the dataset also contains information about the dimensionality
of the input volumes, voxel size, and any other NIfTI-specific information
since it also includes a dump of the full NIfTI image header.

.. note::
   Previously (0.4.x versions and 0.5 development prior March 03, 2010),
   PyMVPA exposed 4D (and 3D with degenerate 1st dimension) data in ``tkji``
   (corresponds to ``tzyx`` if volumes were axial slices in
   neurologic convention) order of dimensions.  Now it uses more convenient
   order ``tijk`` (corresponding to ``txyz``), which will match the order exposed
   by NiBabel (PyNIfTI and NiftiImage still expose them as ``tkji``).

>>> ds.sa.time_indices[:5]
array([0, 1, 2, 3, 4])
>>> ds.sa.time_coords[:5]
array([  0. ,   2.5,   5. ,   7.5,  10. ])
>>> ds.fa.voxel_indices[:5]
array([[ 6, 23, 24],
       [ 7, 18, 25],
       [ 7, 18, 26],
       [ 7, 18, 27],
       [ 7, 19, 25]])
>>> ds.a.voxel_eldim
(3.5, 3.75, 3.75)
>>> ds.a.voxel_dim
(40, 64, 64)
>>> 'imghdr' in ds.a
True

In addition to all this information, the dataset also carries a key
attribute: the *mapper*. A mapper is an important concept in PyMVPA, and
hence worth devoting the whole :ref:`next tutorial chapter
<chap_tutorial_mappers>` to it.

>>> print ds.a.mapper
<Chain: <Flatten>-<StaticFeatureSelection>>

Having all these attributes being part of a dataset is often a useful thing
to have, but in some cases (e.g. when it comes to efficiency, and/or very
large datasets) one might want to have a leaner dataset with just the
information that is really necessary. One way to achieve this, is to strip
all unwanted attributes. The Dataset class'
:meth:`~mvpa.base.dataset.AttrDataset.copy()` method can help with that.

>>> stripped = ds.copy(deep=False, sa=['time_coords'], fa=[], a=[])
>>> print stripped
<Dataset: 1452x577@int16, <sa: time_coords>>

We can see that all attributes besides ``time_coords`` have been filtered out.
Setting the ``deep`` arguments to ``False`` causes the copy function to reuse the
data from the source dataset to generate the new stripped one, without
duplicating all data in memory -- meaning both datasets now share the sample
data and any change done to ``ds`` will also affect ``stripped``.


Storage
=======

Some data preprocessing can take a long time.  One would rather prevent
doing it over and over again, and instead just store the preprocessed data
into a file for subsequent analyses. PyMVPA offers functionality to store a
large variety of objects, including datasets, into HDF5_ files. A variant
of this format is also used by recent versions of Matlab to store data.

.. _HDF5: http://en.wikipedia.org/wiki/Hierarchical_Data_Format
.. _h5py: http://h5py.alfven.org

For HDF5 support PyMVPA depends on the h5py_ package. If it is available,
any dataset can be saved to a file by simply calling
`~mvpa.base.dataset.AttrDataset.save()` with the desired filename.

>>> import tempfile, shutil
>>> # create a temporary directory
>>> tempdir = tempfile.mkdtemp()
>>> ds.save(os.path.join(tempdir, 'mydataset.hdf5'))

HDF5 is a flexible format that also supports, for example, data
compression. To enable it, you can pass additional arguments to
`~mvpa.base.dataset.AttrDataset.save()` that are supported by
`Group.create_dataset()`. Instead of using
`~mvpa.base.dataset.AttrDataset.save()` one can also use the `~mvpa.base.hdf5.h5save()`
function in a similar way. Saving the same dataset with maximum
gzip-compression looks like this:

>>> ds.save(os.path.join(tempdir, 'mydataset.gzipped.hdf5'), compression=9)
>>> h5save(os.path.join(tempdir, 'mydataset.gzipped.hdf5'), ds, compression=9)

Loading datasets from a file is easy too. `~mvpa.base.hdf5.h5load()` takes a filename as
an argument and returns the stored dataset. Compressed data will be handled
transparently.

>>> loaded = h5load(os.path.join(tempdir, 'mydataset.hdf5'))
>>> np.all(ds.samples == loaded.samples)
True
>>> # cleanup the temporary directory, and everything it includes
>>> shutil.rmtree(tempdir, ignore_errors=True)
