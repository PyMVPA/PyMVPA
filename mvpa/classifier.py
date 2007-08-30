### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    Abstract base class for all classifiers.
#
#    Copyright (C) 2007 by
#    Michael Hanke <michael.hanke@gmail.com>
#
#    This package is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 2 of the License, or (at your option) any later version.
#
#    This package is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import operator

class Classifier(object):
    def __init__(self, patterns, capabilities):
        """
          Parameters:
            data         - MVPAPattern instance with the training data
            capabilities - List of strings with capability labels (see below)

          List of classifier capability labels:
            feature_benchmarks - implements getFeatureBenchmark()
        """

        if not operator.isSequenceType( capabilities ):
            raise ValueError, 'capabilities has to be a sequence'

        self.__capabilities = capabilities
        self.__patterns = patterns


    def train(self):
        raise NotImplementedError


    def predict(self, data):
        raise NotImplementedError


    def getFeatureBenchmark(self):
        raise NotImplementedError


    # read-only properties
    capabilities = property( fget=lambda self: self.__capabilities )
    patterns = property( fget=lambda self: self.__patterns )
