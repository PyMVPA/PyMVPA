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

Most of the time a dataset will hold its samples in a NumPy array. However, we
have already seens that not only arrays can be used to create a dataset (e.g.
the first example passed the samples as a nested list).  Actually, the dataset
implementation supports multiple samples container type (benfitting from Python
being a dynamically typed programming language. It follows a simple rules what
can be stored:

* If the samples are passed as a list, it is converted into a NumPy array.
* All other objects are tested whether they comply with two criteria:

   a. It must have a `dtype` attribute that reports the datatype
      of the samples in a way that is compatible with the NumPy
      array interface.
   b. It must have a `shape` attribute that behave similar to that of NumPy
      arrays *and* the reported shape must indicate at least one present axis
      (i.e. so-called zero-dim array are not supported).

If the above conditions are verified, one-dimensional data is converted into a
two-dimensional array, by considering all data as a single feature and multiple
samples. Otherwise all datatypes that fulfill these conditions can serve as a
samples container inside a dataset. However, some useful functionality provided
by a dataset might add additional requirements, and hence, will be unavailable
with incompatible containers. Most popular alternatives to plain NumPy arrays
are NumPy matrices, SciPy's sparse matrices, and custom ndarray subclasses. All
of these examples should work with a dataset.


Slicing
=======


Attributes
==========

For Samples
-----------

For Features
------------

For The Dataset
---------------

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
