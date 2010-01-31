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

General Changes
===============

Datasets
========

Interface changes
-----------------

Behaves more like a NumPy array.


Sparse data support
-------------------

Dataset in principal now support non-ndarray types for dataset samples. However,
most parts of PyMVPA still assume an (at least) ndarray-like interface.


Classifiers
===========
