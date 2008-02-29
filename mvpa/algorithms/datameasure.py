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
from mvpa.clfs.classifier import BoostedClassifier, ProxyClassifier
from mvpa.clfs.svm import LinearSVM
from mvpa.misc.transformers import Absolute, Identity, FirstAxisMean

if __debug__:
    from mvpa.misc import debug

class DatasetMeasure(Stateful):
    """A measure computed from a `Dataset` (base class).

    All subclasses shall get all necessary parameters via their constructor,
    so it is possible to get the same type of measure for multiple datasets
    by passing them to the __call__() method successively.
    """

    raw_result = StateVariable(enabled=False,
        doc="Result before applying transformer")

    def __init__(self, transformer=None, *args, **kwargs):
        """Does nothing special.

        :Parameter:
            transformer: Functor
                This functor is called in `__call__()` to perform a final
                processing step on the to be returned dataset measure. If None,
                nothing is called
        """
        Stateful.__init__(self, **kwargs)

        self.__transformer = transformer
        """Functor to be called in return statement of all subclass __call__()
        methods."""


    def __call__(self, dataset):
        """Compute measure on a given `Dataset`.

        Each implementation has to handle a single arguments: the source
        dataset.

        Returns the computed measure in some iterable (list-like)
        container applying transformer if such is defined
        """
        result = self._call(dataset)
        self.raw_result = result
        if not self.__transformer is None:
            result = self.__transformer(result)
        return result


    def _call(self, dataset):
        """Actually compute measure on a given `Dataset`.

        Each implementation has to handle a single arguments: the source
        dataset.

        Returns the computed measure in some iterable (list-like) container.
        """
        raise NotImplemented



class ScalarDatasetMeasure(DatasetMeasure):
    """A scalar measure computed from a `Dataset` (base class).

    Should behave like a DatasetMeasure.
    """
    def __init__(self, *args, **kwargs):
        """Does nothing."""
        DatasetMeasure.__init__(self, *(args), **(kwargs))


    def _call(self, dataset):
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


    def _call(self, dataset):
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


    def _call(self, dataset):
        """Perform sensitivity analysis on a given `Dataset`.

        Each implementation has to handle a single arguments: the source
        dataset.

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


    def __call__(self, dataset):
        """Train classifier on `dataset` and then compute actual sensitivity.
        """
        if not self.clf.trained or self._force_training:
            if __debug__:
                debug("SA", "Training classifier %s %s" %
                      (`self.clf`,
                       {False: "since it wasn't yet trained",
                        True:  "although it was trained previousely"}
                       [self.clf.trained]))
            self.clf.train(dataset)

        return SensitivityAnalyzer.__call__(self, dataset)


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
        banalyzer = LinearSVMWeights(clf, transformer=Absolute, **kwargs)
    elif isinstance(clf, BoostedClassifier):
        if basic_analyzer is None and len(clf.clfs) > 0:
            basic_analyzer = selectAnalyzer(clf.clfs[0], **kwargs)
            if __debug__:
                debug("SA", "Selected basic analyzer %s for classifier %s " +
                      "based on 0th classifier in it being %s" %
                      (analyzer, clf, clf.clfs[0] ))
        banalyzer = BoostedClassifierSensitivityAnalyzer(clf,
                            analyzer=basic_analyzer, **kwargs)
    elif isinstance(clf, ProxyClassifier):
        if basic_analyzer is None:
            basic_analyzer = selectAnalyzer(clf.clf, **kwargs)
            if __debug__:
                debug("SA", "Selected basic analyzer %s for classifier %s " %
                      (basic_analyzer, clf) +
                      "based on proxied classifier in it being %s" % clf.clf)
        banalyzer = ProxyClassifierSensitivityAnalyzer(clf,
                                                       analyzer=basic_analyzer,
                                                       **kwargs)

    return banalyzer


class CombinedSensitivityAnalyzer(SensitivityAnalyzer):
    """Set sensitivity analyzers to be merged into a single output"""

    sensitivities = StateVariable(enabled=False,
        doc="Sensitivities produced by each classifier")

    def __init__(self, analyzers=None,
                 combiner=FirstAxisMean,
                 **kwargs):
        if analyzers == None:
            analyzers = []

        SensitivityAnalyzer.__init__(self, **kwargs)
        self.__analyzers = analyzers
        """List of analyzers to use"""

        self.__combiner = combiner
        """Which functor to use to combine all sensitivities"""



    def _call(self, dataset):
        sensitivities = []
        ind = 0
        for analyzer in self.__analyzers:
            if __debug__:
                debug("SA", "Computing sensitivity for SA#%d:%s" %
                      (ind, analyzer))
            sensitivity = analyzer(dataset)
            sensitivities.append(sensitivity)
            ind += 1

        self.sensitivities = sensitivities
        return self.__combiner(sensitivities)


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


    def _call(self, dataset):
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

        return self.__combined_analyzer(dataset)

    combined_analyzer = property(fget=lambda x:x.__combined_analyzer)


class ProxyClassifierSensitivityAnalyzer(ClassifierBasedSensitivityAnalyzer):
    """Set sensitivity analyzer output just to pass through"""

    def __init__(self,
                 clf,
                 analyzer=None,
                 **kwargs):
        """Initialize Sensitivity Analyzer for `BoostedClassifier`
        """
        ClassifierBasedSensitivityAnalyzer.__init__(self, clf, **kwargs)

        self.__analyzer = None
        """Analyzer to use for basic classifiers within boosted classifier"""


    def _call(self, dataset):
        if self.__analyzer is None:
            self.__analyzer = selectAnalyzer(self.clf.clf)
            if self.__analyzer is None:
                raise ValueError, \
                      "Wasn't able to figure basic analyzer for clf %s" % \
                      `clf`
            if __debug__:
                debug("SA", "Selected analyzer %s for clf %s" % \
                      (`self.__analyzer`, `self.clf.clf`))

        # TODO "remove" unnecessary things below on each call...
        # assign corresponding classifier
        self.__analyzer.clf = self.clf.clf

        # if clf was trained already - don't train again
        if self.clf.clf.trained:
            self.__analyzer._force_training = False

        return self.__analyzer._call(dataset)

    analyzer = property(fget=lambda x:x.__analyzer)
