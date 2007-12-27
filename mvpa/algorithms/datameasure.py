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

from mvpa.misc.state import State

if __debug__:
    from mvpa.misc import debug

class DataMeasure(State):
    """A measure computed from a `Dataset` (base class).

    All subclasses shall get all necessary parameters via their constructor,
    so it is possible to get the same type of measure for multiple datasets
    by passing them to the __call__() method successively.
    """

    def __init__(self, **kwargs):
        """
        """
        State.__init__(self, **kwargs)


    def __call__(self, dataset, callbacks=[]):
        """Compute measure on a given `Dataset`.

        Each implementation has to handle two arguments. The first is the
        source dataset and the second is a list of callables which have to be
        called with the result of the computation.

        Returns the computed measure in some iterable (list-like) container.
        """
        raise NotImplementedError



class SensitivityAnalyzer(DataMeasure):
    """Base class of all sensitivity analysers.

    A sensitivity analyser is an algorithm that assigns a sensitivity value to
    all features in a dataset.
    """
    def __init__(self, **kwargs):
        """Does nothing special."""
        DataMeasure.__init__(self, **kwargs)


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
        # XXX should we allow to return multiple maps (as a sequence) as
        # currently (illegally) done by SplittingSensitivityAnalyzer?
        raise NotImplementedError


#
# Flavored implementations of SensitivityAnalyzers

class ClassifierBasedSensitivityAnalyzer(SensitivityAnalyzer):

    def __init__(self, clf, force_training=True, **kwargs):
        """Initialize the analyzer with the classifier it shall use.

        :Parameters:
          clf : Classifier
            classifier to use. Only classifiers sub-classed from
            `LinearSVM` may be used.
          force_training : Bool
            if classifier was already trained -- do not retrain
        """

        """Does nothing special."""
        SensitivityAnalyzer.__init__(self, **kwargs)

        self.__clf = clf
        """Classifier used to computed sensitivity"""

        self._force_training = force_training
        """Either to force it to train"""


    def __repr__(self):
        return "<ClassifierBasedSensitivityAnalyzer on %s. force_training=%s" % \
               (`self.__clf`, str(force_training))


    def __call__(self, dataset, callables=[]):
        """Train linear SVM on `dataset` and extract weights from classifier.
        """
        if not self.clf.trained or self._force_training:
            if __debug__:
                debug("SA", "Training classifier %s %s" %
                      (`self.clf`,
                       {False: "since it wasn't yet trained",
                        True:  "although it was trained previousely"}
                       [self.clf.trained]))
            self.clf.train(dataset)

        return self._call(dataset, callables)


    def _call(self, dataset, callables=[]):
        """Actually the function which does the computation"""
        raise NotImplementedError

    clf = property(fget=lambda self:self.__clf)
