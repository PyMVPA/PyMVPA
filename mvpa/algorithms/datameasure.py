#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: base class for data measures, algorithms that quantify properties of
           datasets."""


class DataMeasure(object):
    """
    All subclasses shall get all necessary parameters via their constructor,
    so it is possible to get the same type of measure for multiple datasets
    by passing them to the __call__() method successively.
    """
    def __call(self, dataset, callbacks=[]):
        """
        Each implementation has to handle two arguments. The first is the
        source dataset and the second is a list of callables which have to be
        called with the result of the computation.

        Returns the computed measure in some iterable (list-like) container.
        """
        raise NotImplementedError



class SensitivityAnalyzser(DataMeasure):
    """ Base class of all sensitivity analysers.

    A sensitivity analyser is an algorithm that assigns a sensitivity value to
    all features in a dataset.
    """
    def __call(self, dataset, callbacks=[]):
        """
        Each implementation has to handle two arguments. The first is the
        source dataset and the second is a list of callables which have to be
        called with the result of the computation.

        Returns the computed sensitivity measure in a 1D array which's length
        and order matches the features in the dataset. Higher sensitivity values
        should indicate higher sensitivity (or signal to noise ratio or
        amount of available information or the like).
        """
        raise NotImplementedError
