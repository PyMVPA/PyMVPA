# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Algorithms for (reversible) data transformation.

Mappers are objects than can take data, either in a `Dataset`, or plain data
arrays, and transform them in some specific way. There are no general
limitations on what this transformation might be. it can be as simple as
selecting a subset of data (e.g. `StaticFeatureSelection`) or more complex
data projection (e.g. `PCAMapper`).

Mapping algorithms might be unsupervised or supervised techniques, and any
mapper might implement a training step that has to be run before it can be used.

.. note::

  Classifiers from the `mvpa2.clfs` module could also be considered mappers as
  well, but they all are supervised, and only provide ND->1D mapping (from data
  samples onto the target labels), most of the time without the possibility for
  reverse transformation.


Modes Of Operation
==================

Training
  All mappers can be trained by passing a training dataset to their
  :meth:`~mvpa2.mappers.base.Mapper.train()` method. Mappers that do not need to
  be trained will silently ignore this call. Mappers do not modify training
  datasets.

Forward-mapping
  The mapper takes a dataset (or plain data array), transforms it, and returns
  a new dataset (or data array). Mappers follow a copy-on-write (COW) scheme that
  only changes/copies data that is modified by the mapper -- all other
  information will be shared by input and output dataset. If this behavior is
  not appropriate in a particular case, the input dataset should be copied
  manually and only the copy should be given to the mapper.

  Forward-mapping is possible via two different methods:
  `~mvpa2.mappers.base.Mapper.forward()` takes either a `Dataset`, or an at least
  two-dimensional data array. In the latter case, the first axis is assumed to
  separate between samples, as in a dataset. The method will return the
  transformation result in the same format: either a dataset, or an array with
  at least two dimensions.  `~mvpa2.mappers.base.Mapper.forward1()` on the other
  hand only takes plain data arrays that have to be of the same shape as a
  *single* sample in the dataset that the mapper has been trained on. It will
  also return a plain data array.

Reverse-mapping
  If a mapper supports reversing a transformation, dataset and plain data
  arrays can be reverse-mapped with the corresponding methdod.
  `~mvpa2.mappers.base.Mapper.reverse()` and
  `~mvpa2.mappers.base.Mapper.reverse1()` behave analogous to the respective
  forward-mapping functions, and also have the same requirement for their input
  data.
"""

__docformat__ = 'restructuredtext'

if __debug__:
    from mvpa2.base import debug
    debug('INIT', 'mvpa2.mappers')

# do not pull them all -- we have mvpa2.suite for that
#from mvpa2.mappers.mask import MaskMapper
#from mvpa2.mappers.svd import SVDMapper
#from mvpa2.mappers.boxcar import BoxcarMapper
#from mvpa2.mappers.array import DenseArrayMapper

if __debug__:
    debug('INIT', 'mvpa2.mappers end')
