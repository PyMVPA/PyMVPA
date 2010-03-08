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
  NumPy's `ndarrays`, e.g. sparse matrices.


Critical API Changes
====================

* `.states` -> `.ca` (for conditional attributes).  All attributes stored in
  collections (parameters for Classifiers in `.params`, states in `.ca`)
  should be accessed not at top level of the object but through a collection.

* Dataset: behaves more like a NumPy array.  No specialized Dataset classes,
  but constructors
  - MaskedDataset -> `dataset_wizard`
  - NiftiDataset -> `fmri_dataset`
  - ERNiftiDataset -> `fmri_dataset` + call to `.extract_events`

* MRI volumes: 3,4D volumes (and coordinates) are exposed with following order
  of axes: t,x,y,z.  Previously we followed a convention of t,z,y,x order of
  axis in volume data (to be consistent with PyNIfTI).

* Masks (`mask_mapper`)
 - now [1,1,0] is not the same as [True, True, False]

* We have weird (but consistent) conventions now
  - classes are CamelCased
  - factory functions (even for whatever might have been before a class)
    are in pythonic_style

* `detrend` -> `poly_detrend`

* ``perchunk=bool`` (in zscore/detrend) got refactored into ``chunks_attr=None
  or string`` to specify on which sample attribute to operate.

* internally and as provided by mvpa.suite, `numpy` is imported as `np`, and
  `pylab` is imported as `pl`

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


Classifiers
===========
