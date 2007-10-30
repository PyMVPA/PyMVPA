#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: Abstract base class for all classifiers."""


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

    def __init__(self):
        """
        """

    def setProperty(self, propName, propValue):
        raise NotImplementedError

    def train(self, data):
        raise NotImplementedError


    def predict(self, data):
        raise NotImplementedError


    def getFeatureBenchmark(self):
        raise NotImplementedError
