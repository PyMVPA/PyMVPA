# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Data mapper"""

__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.base.dochelpers import enhancedDocString
from mvpa.mappers.base import Mapper
from mvpa.misc.transformers import FirstAxisMean

if __debug__:
    from mvpa.base import debug


class SampleGroupMapper(Mapper):
    # name is ugly, please help!
    """Mapper to apply a mapping function to samples of the same type.

    A customimzable function is applied individually to all samples with the
    same unique label from the same chunk. This mapper is somewhat
    unconventional since it doesn't preserve number of samples (ie the size of
    0-th dimension...)
    """

    def __init__(self, fx=FirstAxisMean):
        """Initialize the PCAMapper

        Parameters:
          startpoints:    A sequence of index value along the first axis of
                          'data'.
          boxlength:      The number of elements after 'startpoint' along the
                          first axis of 'data' to be considered for averaging.
          offset:         The offset between the starting point and the
                          averaging window (boxcar).
          collision_resolution : string
            if a sample belonged to multiple output samples, then on reverse,
            how to resolve the value (choices: 'mean')
        """
        Mapper.__init__(self)

        self.__fx = fx
        self.__uniquechunks = None
        self.__uniquelabels = None
        self.__chunks = None
        self.__labels = None
        self.__datashape = None


    __doc__ = enhancedDocString('SampleGroupMapper', locals(), Mapper)


    def train(self, dataset):
        """
        """
        # just store the relevant information
        self.__uniquechunks = dataset.uniquechunks
        self.__uniquelabels = dataset.uniquelabels
        self.__chunks = dataset.chunks
        self.__labels = dataset.labels
        self.__datashape = (dataset.nfeatures, )


    def forward(self, data):
        """
        """
        if self.__datashape is None:
            raise RuntimeError, \
                  "SampleGroupMapper needs to be trained before it can be used"

        mdata = []

        # for each label in each chunk
        for c in self.__uniquechunks:
            for l in self.__uniquelabels:
                mdata.append(self.__fx(data[N.logical_and(self.__labels == l,
                                                          self.__chunks == c)]))

        return N.array(mdata)


    def reverse(self, data):
        """This is not implemented."""
        raise NotImplementedError


    def getInSize(self):
        """Returns the number of original samples which were combined.
        """
        return self.__datashape[0]


    def getOutSize(self):
        """Returns the number of output samples.
        """
        return self.__datashape[0]


    def selectOut(self, outIds):
        """Just complain for now"""
        raise NotImplementedError, \
            "For feature selection use MaskMapper on output of the %s mapper" \
            % self.__class__.__name__



