### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: Mapped Dataset
#
#    Copyright (C) 2007 by
#    Michael Hanke <michael.hanke@gmail.com>
#
#    This package is free software; you can redistribute it and/or
#    modify it under the terms of the MIT License.
#
#    This package is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the COPYING
#    file that comes with this package for more details.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from dataset import *

class MappedDataset(Dataset, Mapper):
    """
    """
    def __init__(self, samples, labels, chunks, mapper ):
        """
        """
        # store the mapper and put the rest into the baseclass
        Dataset.__init__(self, samples, labels, chunks)

        if not self.nfeatures == mapper.nfeatures:
            raise ValueError, "The mapper doesn't match the number of " \
                              "features in the samples array."
        self.__mapper = mapper


    def __iadd__( self, other ):
        """
        Warning: the current mapper is kept!
        """
        # call base class method to merge the samples
        Dataset.__iadd__(self, other)

        return self


    def __add__( self, other ):
        """
        When adding two MappedDatasets the mapper of the dataset left of the
        operator is used for the merged dataset.
        """
        out = MappedDataset( self.samples,
                             self.labels,
                             self.chunks,
                             self.mapper )

        out += other

        return out


    def setMapper( self, mapper ):
        """
        """
        # if the new mapper operates on a different number of features
        # this class does not know howto handle that
        if not self.mapper.nfeatures == mapper.nfeatures:
            raise ValueError, 'New mapper has to operate on the same number ' \
                              'of features as the old one.'

        self.__mapper = mapper


    def forward(self, data):
        """ Map data from the original dataspace into featurespace.
        """
        return self.__mapper.forward(data)


    def reverse(self, data):
        """ Reverse map data from featurespace into the original dataspace.
        """
        return self.__mapper.reverse(data)


    def selectSamples( self, mask ):
        """ Choose a subset of samples.

        Returns a new MappedDataset object containing the selected sample
        subset.
        """
        # without having a sequence a index the masked sample array would
        # loose its 2d layout
        if not operator.isSequenceType( mask ):
            mask = [mask]

        # XXX should be generic...
        return MappedDataset( self.samples[mask,],
                              self.labels[mask,],
                              self.chunks[mask,],
                              self.mapper )


    # read-only class properties
    mapper = property( fget=lambda self: self.__mapper )
