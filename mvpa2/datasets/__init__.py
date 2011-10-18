# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Data containers and utility functions

Virtually any processing done with PyMVPA involves datasets -- the primary form
of data representation in PyMVPA. Datasets serve as containers for input data,
as well as the return datatype of more complex PyMVPA algorithms.

Most of the time a dataset will hold its samples in a NumPy array. However,
we have already seen that not only arrays can be used to create a dataset
(e.g.  the first example passed the samples as a nested list).  Actually,
the dataset implementation supports multiple samples container types
(benefitting from Python being a dynamically typed programming language). It
follows a simple rule to decide what can be stored:

* If samples are passed as a list, it is converted into a NumPy array.
* All other objects are tested whether they comply with two criteria:

   a. It must have a `dtype` attribute that reports the datatype
      of the samples in a way that is compatible with the NumPy
      array interface.
   b. It must have a `shape` attribute that behave similar to that of NumPy
      arrays *and* the reported shape must indicate at least one present axis
      (i.e. so-called zero-dim arrays are not supported).

If the above conditions are verified, one-dimensional data is converted into a
two-dimensional array, by considering all data as multiple samples
with a single feature. Otherwise all datatypes that fulfill these conditions
can serve as a samples container inside a dataset. However, some useful
functionality provided by a dataset might add additional requirements, and
hence, will be unavailable with incompatible containers. Most popular
alternatives to plain NumPy arrays are NumPy matrices, SciPy's sparse matrices,
and custom ndarray subclasses. All of these examples should work with a
dataset. It should be noted that the samples container is stored *as-is* in the
dataset (unless it was a list that got converted into an array):

  >>> from mvpa2.suite import *
  >>> import scipy.sparse as sparse
  >>> mat = sparse.csc_matrix((10000, 20000))
  >>> sparse_ds = Dataset(mat)
  >>> type(sparse_ds.samples)
  <class 'scipy.sparse.csc.csc_matrix'>
  >>> len(sparse_ds)
  10000
  >>> sparse_ds.nfeatures
  20000


"""

__docformat__ = 'restructuredtext'

if __debug__:
    from mvpa2.base import debug
    debug('INIT', 'mvpa2.datasets')

# nothing in here that works without the base class
from mvpa2.datasets.base import Dataset, dataset_wizard
from mvpa2.base.dataset import hstack, vstack

if __debug__:
    debug('INIT', 'mvpa2.datasets end')
