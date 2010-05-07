.. -*- mode: rst; fill-column: 78; indent-tabs-mode: nil -*-
.. ex: set sts=4 ts=4 sw=4 et tw=79:
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
  #
  #   See COPYING file distributed along with the PyMVPA package for the
  #   copyright and license terms.
  #
  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


.. index:: release notes
.. _chap_release_notes_0.5:

***************************
Release Notes -- PyMVPA 0.5
***************************

For The Impatient
=================

* Datasets are no longer relatively static objects, but become flexible
  multi-purpose containers that can handle attributes for samples, feature,
  or whole datasets. There is some inital support for other datatypes than
  NumPy's ``ndarrays``, e.g. sparse matrices.

* All mappers, classifiers, regressions, and measure are now implemented as
  `Node`\s that can be called with a `Dataset` and return a processed dataset.
  All nodes provide a ``generat()`` method that causes the node to yields to
  processing result. In addition, there is special nodes that implement more
  complex generators yielding multiple results (e.g. resampling, permuting, or
  splitting nodes).


Critical API Changes
====================

* ``.states`` -> ``.ca`` (for conditional attributes).  All attributes stored in
  collections (parameters for Classifiers in ``.params``, states in ``.ca``)
  should be accessed not at top level of the object but through a collection.
  For example: ``clf.confusion`` becomes ``clf.ca.confusion``

* Dataset: behaves more like a NumPy array.  No specialized Dataset classes,
  but factory functions:

  - MaskedDataset -> `dataset_wizard`
  - NiftiDataset -> `fmri_dataset`
  - ERNiftiDataset -> `fmri_dataset` + `eventrelated_dataset` (see
    :ref:`event-related analysis example <example_eventrelated_>`)

* MRI volumes: 3,4D volumes (and coordinates) are exposed with following order
  of axes: t,x,y,z.  Previously we followed a convention of t,z,y,x order of
  axis in volume data (to be consistent with PyNIfTI).

* Masks (`mask_mapper`)

 - now ``[1,1,0]`` is not the same as ``[True, True, False]``

   This first is a list of indices, and the second is a boolean selection
   vector.

* We have weird (but consistent) conventions now
  - classes are CamelCased
  - factory functions (even for whatever might have been before a class)
    are lowercase with underscores

* `detrend` -> `poly_detrend`

* ``perchunk=bool`` (in zscore/detrend) got refactored into ``chunks_attr=None
  or string`` to specify on which sample attribute to operate.

* internally and as provided by mvpa.suite, `numpy` is imported as `np`, and
  `pylab` is imported as `pl`

* Calling mappers (`Mapper.__call__()`) is no longer a simple alias to
  `Mapper.forward()`, but instead causes a mapper to behave like a generator and
  yield (potentially multiple) results when calling with a single input dataset.


General Changes
===============

Datasets
========

Interface changes
-----------------


Sparse data support
-------------------

Dataset in principal now support non-ndarray types for dataset samples. However,
most parts of PyMVPA still assume an (at least) ndarray-like interface.

Splitters
---------

* `permute` -> `permute_attr`, so if you had `permute=True`, use
  `attr='targets'` if you like to permute targets


Classifiers
===========
