### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    Abstract base class for all classifiers.
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

import operator

class Classifier(object):
    """
    Required behavior:
    
    For every classifier is has to be possible to be instanciated without
    having to specify the training pattern.

    Repeated calls to the train() method with different training data have to
    result in a valid classifier, trained for the particular dataset.

    It must be possible to specify all classifier parameters as keyword arguments
    to the constructor.
    """
    def __init__(self, capabilities):
        """
          Parameters:
            capabilities - List of strings with capability labels (see below)

          List of classifier capability labels:
            feature_benchmarks - implements getFeatureBenchmark()
        """

        if not operator.isSequenceType( capabilities ):
            raise ValueError, 'capabilities has to be a sequence'

        self.__capabilities = capabilities


    def train(self, data):
        raise NotImplementedError


    def predict(self, data):
        raise NotImplementedError


    def getFeatureBenchmark(self):
        raise NotImplementedError


    # read-only properties
    capabilities = property( fget=lambda self: self.__capabilities )
