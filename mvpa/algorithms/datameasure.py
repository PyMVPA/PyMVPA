#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Base class for data measures: algorithms that quantify properties of
datasets.

Besides the `DataMeasure` base class this module also provides the (abstract)
`SensitivityAnalyzer` class. The difference between a general measure and
the output of the `SensitivityAnalyzer` is that the latter returns a 1d map
(one value per feature in the dataset). In contrast there are no restrictions
on the returned value of `DataMeasure` except for that it has to be in some
iterable container.
"""

__docformat__ = 'restructuredtext'

class DataMeasure(object):
    """A measure computed from a `Dataset` (base class).

    All subclasses shall get all necessary parameters via their constructor,
    so it is possible to get the same type of measure for multiple datasets
    by passing them to the __call__() method successively.
    """
    def __call__(self, dataset, callbacks=[]):
        """Compute measure on a given `Dataset`.

        Each implementation has to handle two arguments. The first is the
        source dataset and the second is a list of callables which have to be
        called with the result of the computation.

        Returns the computed measure in some iterable (list-like) container.
        """
        raise NotImplementedError



class SensitivityAnalyzer(DataMeasure):
    """ Base class of all sensitivity analysers.

    A sensitivity analyser is an algorithm that assigns a sensitivity value to
    all features in a dataset.
    """
    def __call__(self, dataset, callbacks=[]):
        """Perform sensitivity analysis on a given `Dataset`.

        Each implementation has to handle two arguments. The first is the
        source dataset and the second is a list of callables which have to be
        called with the result of the computation.

        Returns the computed sensitivity measure in a 1D array which's length
        and order matches the features in the dataset. Higher sensitivity values
        should indicate higher sensitivity (or signal to noise ratio or
        amount of available information or the like).
        """
        raise NotImplementedError
