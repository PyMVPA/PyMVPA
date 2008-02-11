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

Besides the `DatasetMeasure` base class this module also provides the (abstract)
`SensitivityAnalyzer` class. The difference between a general measure and
the output of the `SensitivityAnalyzer` is that the latter returns a 1d map
(one value per feature in the dataset). In contrast there are no restrictions
on the returned value of `DatasetMeasure` except for that it has to be in some
iterable container.
"""

__docformat__ = 'restructuredtext'

import numpy as N
import copy

from mvpa.misc.state import StateVariable, Stateful
from mvpa.clfs.classifier import BoostedClassifier
from mvpa.clfs.svm import LinearSVM

if __debug__:
    from mvpa.misc import debug

class DatasetMeasure(Stateful):
    """A measure computed from a `Dataset` (base class).

    All subclasses shall get all necessary parameters via their constructor,
    so it is possible to get the same type of measure for multiple datasets
    by passing them to the __call__() method successively.
    """
    def __init__(self, *args, **kwargs):
        """Does nothing."""
        Stateful.__init__(self, **kwargs)


    def __call__(self, dataset, callbacks=[]):
        """Compute measure on a given `Dataset`.

        Each implementation has to handle two arguments. The first is the
        source dataset and the second is a list of callables which have to be
        called with the result of the computation.

        Returns the computed measure in some iterable (list-like) container.
        """
        raise NotImplementedError



class ScalarDatasetMeasure(DatasetMeasure):
    """A scalar measure computed from a `Dataset` (base class).

    Should behave like a DatasetMeasure.
    """
    def __init__(self, *args, **kwargs):
        """Does nothing."""
        DatasetMeasure.__init__(self, *(args), **(kwargs))


    def __call__(self, dataset, callbacks=[]):
        """Computes a scalar measure on a given `Dataset`.

        Behaves like a `DatasetMeasure`, but computes and returns a single
        scalar value.
        """
        raise NotImplementedError



class FeaturewiseDatasetMeasure(DatasetMeasure):
    """A per-feature-measure computed from a `Dataset` (base class).

    Should behave like a DatasetMeasure.
    """
    def __init__(self, *args, **kwargs):
        """Does nothing."""
        DatasetMeasure.__init__(self, *(args), **(kwargs))


    def __call__(self, dataset, callbacks=[]):
        """Computes a per-feature-measure on a given `Dataset`.

        Behaves like a `DatasetMeasure`, but computes and returns a 1d ndarray
        with one value per feature.
        """
        raise NotImplementedError



class SensitivityAnalyzer(FeaturewiseDatasetMeasure):
    """Base class of all sensitivity analysers.

    A sensitivity analyser is an algorithm that assigns a sensitivity value to
    all features in a dataset.
    """
    def __init__(self, *args, **kwargs):
        """Does nothing."""
        FeaturewiseDatasetMeasure.__init__(self, *(args), **(kwargs))


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
        return \
            "<ClassifierBasedSensitivityAnalyzer on %s. force_training=%s" % \
               (`self.__clf`, str(self._force_training))


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


    def _setClassifier(self, clf):
        self.__clf = clf

    clf = property(fget=lambda self:self.__clf,
                   fset=_setClassifier)


def selectAnalyzer(clf, basic_analyzer=None, **kwargs):
    """Factory method which knows few classifiers and their sensitivity
    analyzers.

    :Parameters:
      clf : Classifier
        the one for which to select analyzer
      basic_analyzer : SensitivityAnalyzer
        in case if `clf` is a classifier which uses other classifiers
        specify which basic_analyzer to use when constructing combined analyzer

    This function is just to assign default values. For
    advanced/controlled computation assign them explicitely
    """
    banalyzer = None
    if isinstance(clf, LinearSVM):
        from linsvmweights import LinearSVMWeights
        banalyzer = LinearSVMWeights(clf, **kwargs)
    elif isinstance(clf, BoostedClassifier):
        if basic_analyzer is None and len(clf.clfs) > 0:
            basic_analyzer = selectAnalyzer(clf.clfs[0], **kwargs)
            if __debug__:
                debug("SA", "Selected basic analyzer %s for classifier %s " +
                      "based on 0th classifier in it being %s" %
                      (analyzer, clf, clf.clfs[0] ))
        banalyzer = BoostedClassifierSensitivityAnalyzer(clf,
                            analyzer=basic_analyzer, **kwargs)
    return banalyzer


class CombinedSensitivityAnalyzer(SensitivityAnalyzer):
    """Set sensitivity analyzers to be merged into a single output"""

    sensitivities = StateVariable(enabled=False,
        doc="Sensitivities produced by each classifier")

    def __init__(self, analyzers=[],
                 combiner=lambda x:N.mean(x, axis=0),
                 **kwargs):
        SensitivityAnalyzer.__init__(self, **kwargs)
        self.__analyzers = analyzers
        """List of analyzers to use"""

        self.__combiner = combiner
        """Which functor to use to combine all sensitivities"""



    def __call__(self, dataset, callables=[]):
        sensitivities = []
        ind = 0
        for analyzer in self.__analyzers:
            if __debug__:
                debug("SA", "Computing sensitivity for SA#%d:%s" %
                      (ind, analyzer))
            sensitivity = analyzer(dataset, callables)
            sensitivities.append(sensitivity)
            ind += 1

        self.sensitivities = sensitivities
        result = self.__combiner(sensitivities)
        return result


    def _setAnalyzers(self, analyzers):
        """Set the analyzers
        """
        self.__analyzers = analyzers
        """Analyzers to use"""

    analyzers = property(fget=lambda x:x.__analyzers,
                         fset=_setAnalyzers,
                         doc="Used analyzers")



class BoostedClassifierSensitivityAnalyzer(ClassifierBasedSensitivityAnalyzer):
    """Set sensitivity analyzers to be merged into a single output"""

    def __init__(self,
                 clf,
                 analyzer=None,
                 combined_analyzer=None,
                 **kwargs):
        """Initialize Sensitivity Analyzer for `BoostedClassifier`
        """
        ClassifierBasedSensitivityAnalyzer.__init__(self, clf, **kwargs)
        if combined_analyzer is None:
            combined_analyzer = CombinedSensitivityAnalyzer(**kwargs)
        self.__combined_analyzer = combined_analyzer
        """Combined analyzer to use"""

        self.__analyzer = None
        """Analyzer to use for basic classifiers within boosted classifier"""


    def _call(self, dataset, callables=[]):
        analyzers = []
        # create analyzers
        for clf in self.clf.clfs:
            if self.__analyzer is None:
                analyzer = selectAnalyzer(clf)
                if analyzer is None:
                    raise ValueError, \
                          "Wasn't able to figure basic analyzer for clf %s" % \
                          `clf`
                if __debug__:
                    debug("SA", "Selected analyzer %s for clf %s" % \
                          (`analyzer`, `clf`))
            else:
                # shallow copy should be enough...
                analyzer = copy.copy(self.__analyzer)

            # assign corresponding classifier
            analyzer.clf = clf
            # if clf was trained already - don't train again
            if clf.trained:
                analyzer._force_training = False
            analyzers.append(analyzer)

        self.__combined_analyzer.analyzers = analyzers

        return self.__combined_analyzer(dataset, callables)

    combined_analyzer = property(fget=lambda x:x.__combined_analyzer)
