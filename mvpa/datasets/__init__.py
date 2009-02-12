# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA datasets and helper classes such as splitters

Module Description
==================

`Dataset` and derived classes are dedicated to contain the data and
associated information (such as labels, chunk(session) identifiers.

Module Organization
===================

The mvpa.datasets module contains the following modules:

.. packagetree::
   :style: UML

:group Generic Datasets: base mapped masked meta
:group Specialized Datasets: nifti channel eep event
:group Splitters: splitter
:group Miscellaneous: miscfx miscfx_sp


"""

__docformat__ = 'restructuredtext'

if __debug__:
    from mvpa.base import debug
    debug('INIT', 'mvpa.datasets')

# nothing in here that works without the base class
from mvpa.datasets.base import Dataset

if __debug__:
    debug('INIT', 'mvpa.datasets end')
