#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA datasets and helper classes such as mappers, splitters

Module Description
==================

`Dataset` and derived classes are dedicated to contain the data and
associated information (such as labels, chunk(session) identifiers.

Module Organization
===================

The mvpa.datasets module contains the following modules:

.. packagetree::
   :style: UML

:group Datasets: dataset, mappeddataset, maskeddataset, niftidataset
:group Mappers: mapper, maskmapper
:group Metrics: metric
:group Splitters: splitter, nfoldsplitter
:group Miscellaneous TODO: should be gone somewhere else: misc


"""

__docformat__ = 'restructuredtext'

# nothing in here that works without the base class
from mvpa.datasets.base import Dataset
