.. -*- mode: rst; fill-column: 78 -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

.. _chap_tutorial2:
.. index:: Tutorial, Dataset concepts

********************************************
Tutorial Part 2: Dataset Basics and Concepts
********************************************

A `~mvpa.datasets.base.Dataset` is the basic data container in PyMVPA. It
serves as the primary form of input data storage, but also as container for
more complex results returned by some algorithm. In this tutorial part we will
take a look at what a dataset consists of, and how it works.

In the simplest case, a dataset only contains *data* that is a matrix of
numerical values.

  >>> from mvpa.suite import *
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

In the above example, every row vector in the `data` matrix becomes an
observation or :term:`sample` in the dataset, and every column vector
represents an individual variable or :term:`feature`. The concepts of samples
and features are essential for a dataset, hence we take a further, closer look.

The dataset assumes the first axis of the data to be the samples separating
dimension. If the dataset is created using a one-dimensional vector it will
therefore have as many samples as elements in the vector, and only one feature.

  >>> one_d = [ 0, 1, 2, 3 ]
  >>> one_ds = Dataset(one_d)
  >>> one_ds.shape
  (4, 1)

On the other hand, if a dataset is created from multi-dimensional data, only its
second axis represent the features

  >>> import numpy as N
  >>> m_ds = Dataset(N.random.random((3, 4, 2, 3)))
  >>> m_ds.shape
  (3, 4, 2, 3)
  >>> m_ds.nfeatures
  4

In this case we have a dataset with three samples and four features, where each
feature is a 2x3 matrix. In case somebody is wondering now, why not simply each
value in the data array is considered as its own feature (yielding 24 features)
-- stay tuned, as this is going to be of importance later on.

Most of the time a dataset will hold its samples in a NumPy array. However,
we have already seens that not only arrays can be used to create a dataset
(e.g.  the first example passed the samples as a nested list).  Actually,
the dataset implementation supports multiple samples container type
(benfitting from Python being a dynamically typed programming language). It
follows a simple rule to decide what can be stored:

* If the samples are passed as a list, it is converted into a NumPy array.
* All other objects are tested whether they comply with two criteria:

   a. It must have a `dtype` attribute that reports the datatype
      of the samples in a way that is compatible with the NumPy
      array interface.
   b. It must have a `shape` attribute that behave similar to that of NumPy
      arrays *and* the reported shape must indicate at least one present axis
      (i.e. so-called zero-dim arrays are not supported).

If the above conditions are verified, one-dimensional data is converted into a
two-dimensional array, by considering all data as multiple multiple samples
with a single feature. Otherwise all datatypes that fulfill these conditions
can serve as a samples container inside a dataset. However, some useful
functionality provided by a dataset might add additional requirements, and
hence, will be unavailable with incompatible containers. Most popular
alternatives to plain NumPy arrays are NumPy matrices, SciPy's sparse matrices,
and custom ndarray subclasses. All of these examples should work with a
dataset. It should be noted that the samples container is stored *as-is* in the
dataset (unless it was a list that got converted into an array):

  >>> import scipy.sparse as sparse
  >>> mat = sparse.csc_matrix((10000, 20000))
  >>> sparse_ds = Dataset(mat)
  >>> type(sparse_ds.samples)
  <class 'scipy.sparse.csc.csc_matrix'>
  >>> len(sparse_ds)
  10000
  >>> sparse_ds.nfeatures
  20000


Attributes
==========

What we have seen so far does not really warrant the use of a dataset over
a plain array or matrix with samples. However, in the MVPA context we often
need to know more about each samples than just the value of its features.
In the previous tutorial part we have already seen that :term:`target`
values are required for supervised-learning algorithms, and that a dataset
often has to be split based on the origin of specific groups of samples.
For this type of auxiliary information a dataset can also contain three
types of :term:`attribute`\ s: :term:`sample attribute`, :term:`feature
attribute`, and :term:`dataset attribute`.

For Samples
-----------

In a dataset each :term:`sample` can have an arbitrary number of additional
attributes. They are stored as vectors of length of the number of samples
in a collection that is accessible via the `sa` attribute. A collection is
implemented as a standard Python `dict`, and hence adding sample attributes
work identical to adding elements to a dictionary:

  >>> ds.sa['some_attr'] = [ 0, 1, 1, 3 ]
  >>> ds.sa.keys()
  ['some_attr']

However, sample attributes are not directly stored as plain data, but for
various reasons as a so-called `~mvpa.base.collections.Collectable` that in
turn embeds a NumPy array with the actual attribute:

  >>> type(ds.sa['some_attr'])
  <class 'mvpa.base.collections.ArrayCollectable'>
  >>> ds.sa['some_attr'].value
  array([0, 1, 1, 3])

This "complication" is done to be able to extend attributes with additional
functionality that is often needed and can offer significant speed-up of
processing. For example, sample attributes carry a list of there unique
values, that is only computed once (when first requested) and can subsequently
be directly accessed without repeated and expensive searches:

  >>> ds.sa['some_attr'].unique
  array([0, 1, 3])

However, for most interactive use of PyMVPA this type of attribute access
is relatively complicated (a lot to type), therefore collections offer
direct attribute access by name:

  >>> ds.sa.some_attr
  array([0, 1, 1, 3])

The sample attribute collection also aims to preserve data integrity, by
disallowing improper attributes:

.. code-block:: python

  >> ds.sa['invalid'] = 4
  ValueError: ArrayCollectable only takes sequences as value.
  >> ds.sa['invalid'] = [ 1, 2, 3, 4, 5, 6 ]
  ValueError: Collectable 'invalid' with length [6] does not match the required
  length [4] of collection '<SampleAttributesCollection: some_attr>'.

But, as long as the length of the attribute vector matches the number of
samples in the dataset, and the attributes values can be stored in a NumPy
array, any value is allowed. For example, it is perfectly possible and
supported to store literal attributes.

  >>> ds.sa['literal'] = ['one', 'two', 'three', 'four']

For Features
------------

For The Dataset
---------------

Slicing
=======

Loading fMRI
============

References
==========

Literature
----------

Related API Documentation
-------------------------
.. autosummary::
   :toctree:

   ~mvpa.datasets.base.Dataset
   mvpa.datasets.splitters
