# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA mappers.

Module Description
==================

Various space transformations which are intended to map between two
spaces, most of the time both ways, and optionally requiring training.
Classifiers from the mvpa.clfs module could be considered mappers as
well, but they all are supervised, and only provide ND->1D mapping,
most of the time without reverse transformation.

Module Organization
===================

The mvpa.mappers module contains the following modules:

.. packagetree::
   :style: UML

:group Base: base mask metric
:group Specialized: wavelet boxcar svd ica pca samplegroup

"""

__docformat__ = 'restructuredtext'

if __debug__:
    from mvpa.base import debug
    debug('INIT', 'mvpa.mappers')

# do not pull them all -- we have mvpa.suite for that
#from mvpa.mappers.mask import MaskMapper
#from mvpa.mappers.pca import PCAMapper
#from mvpa.mappers.svd import SVDMapper
#from mvpa.mappers.boxcar import BoxcarMapper
#from mvpa.mappers.array import DenseArrayMapper

if __debug__:
    debug('INIT', 'mvpa.mappers end')
