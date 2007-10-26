#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: Abstract base class for all classifiers."""

import operator


class Classifier(object):
    """
    Required behavior:

    For every classifier is has to be possible to be instanciated without
    having to specify the training pattern.

    Repeated calls to the train() method with different training data have to
    result in a valid classifier, trained for the particular dataset.

    It must be possible to specify all classifier parameters as keyword
    arguments to the constructor.
    """

    _params = []

    def __init__(self, property):
        """
          Parameters:
            capabilities - List of strings with capability labels (see below)

          List of classifier capability labels:
            feature_benchmarks - implements getFeatureBenchmark()
        """

        if not operator.isSequenceType( capabilities ):
            raise ValueError, 'capabilities has to be a sequence'

        self.__capabilities = capabilities


    def setProperty(self, propName, propValue):
        raise NotImplementedError

    def train(self, data):
        raise NotImplementedError


    def predict(self, data):
        raise NotImplementedError


    def getFeatureBenchmark(self):
        raise NotImplementedError


    # read-only properties
    capabilities = property( fget=lambda self: self.__capabilities )
