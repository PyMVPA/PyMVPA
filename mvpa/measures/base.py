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

Besides the `DatasetMeasure` base class this module also provides the
(abstract) `FeaturewiseDatasetMeasure` class. The difference between a general
measure and the output of the `FeaturewiseDatasetMeasure` is that the latter
returns a 1d map (one value per feature in the dataset). In contrast there are
no restrictions on the returned value of `DatasetMeasure` except for that it
has to be in some iterable container.

"""

__docformat__ = 'restructuredtext'

import mvpa.misc.copy as copy

from mvpa.misc.state import StateVariable, Stateful
from mvpa.misc.transformers import FirstAxisMean, SecondAxisSumOfAbs
from mvpa.base.dochelpers import enhancedDocString

if __debug__:
    from mvpa.base import debug


class DatasetMeasure(Stateful):
    """A measure computed from a `Dataset`

    All dataset measures support arbitrary transformation of the measure
    after it has been computed. Transformation are done by processing the
    measure with a functor that is specified via the `transformer` keyword
    argument of the constructor. Upon request, the raw measure (before
    transformations are applied) is stored in the `raw_result` state variable.

    Additionally all dataset measures support the estimation of the
    probabilit(y,ies) of a measure under some distribution. Typically this will
    be the NULL distribution (no signal), that can be estimated with
    permutation tests. If a distribution estimator instance is passed to the
    `null_dist` keyword argument of the constructor the respective
    probabilities are automatically computed and stored in the `null_prob`
    state variable.

    :Developer note:
      All subclasses shall get all necessary parameters via their constructor,
      so it is possible to get the same type of measure for multiple datasets
      by passing them to the __call__() method successively.
    """

    raw_result = StateVariable(enabled=False,
        doc="Computed results before applying any " +
            "transformation algorithm")
    null_prob = StateVariable(enabled=True)
    """Stores the probability of a measure under the NULL hypothesis"""

    def __init__(self, transformer=None, null_dist=None, **kwargs):
        """Does nothing special.

        :Parameter:
          transformer: Functor
            This functor is called in `__call__()` to perform a final
            processing step on the to be returned dataset measure. If None,
            nothing is called
          null_dist : instance of distribution estimator
        """
        Stateful.__init__(self, **kwargs)

        self.__transformer = transformer
        """Functor to be called in return statement of all subclass __call__()
        methods."""
        self.__null_dist = null_dist


    __doc__ = enhancedDocString('DatasetMeasure', locals(), Stateful)


    def __call__(self, dataset):
        """Compute measure on a given `Dataset`.

        Each implementation has to handle a single arguments: the source
        dataset.

        Returns the computed measure in some iterable (list-like)
        container applying transformer if such is defined
        """
        result = self._call(dataset)
        result = self._postcall(dataset, result)
        self.raw_result = result
        if not self.__transformer is None:
            if __debug__:
                debug("SA_", "Applying transformer %s" % self.__transformer)
            result = self.__transformer(result)
        return result


    def _call(self, dataset):
        """Actually compute measure on a given `Dataset`.

        Each implementation has to handle a single arguments: the source
        dataset.

        Returns the computed measure in some iterable (list-like) container.
        """
        raise NotImplemented


    def _postcall(self, dataset, result):
        """Some postprocessing on the result
        """
        # estimate the NULL distribution when functor is given
        if not self.__null_dist is None:
            # we need a matching datameasure instance, but we have to disable
            # the estimation of the null distribution in that child to prevent
            # infinite looping.
            measure = copy.copy(self)
            measure.__null_dist = None
            self.__null_dist.fit(measure, dataset)

            # get probability of result under NULL hypothesis if available
            self.null_prob = self.__null_dist.cdf(result)

        return result


    def __repr__(self, prefixes=[]):
        prefixes = prefixes[:]
        if self.__transformer is not None:
            prefixes.append("transformer=%s" % self.__transformer)
        if self.__null_dist is not None:
            prefixes.append("null_dist=%s" % self.__null_dist)
        return super(DatasetMeasure, self).__repr__(prefixes=prefixes)


class FeaturewiseDatasetMeasure(DatasetMeasure):
    """A per-feature-measure computed from a `Dataset` (base class).

    Should behave like a DatasetMeasure.
    """

    base_sensitivities = StateVariable(enabled=False,
        doc="Stores basic sensitivities if the sensitivity " +
            "relies on combining multiple ones")

    # XXX should we may be default to combiner=None to avoid
    # unexpected results? Also rethink if we need combiner here at
    # all... May be combiners should be 'adjoint' with transformer
    def __init__(self, combiner=SecondAxisSumOfAbs, **kwargs):
        """Initialize

        :Parameters:
          combiner : Functor
            The combiner is only applied if the computed featurewise dataset
            measure is more than one-dimensional. This is different from a
            `transformer`, which is always applied. By default, the sum of
            absolute values along the second axis is computed.
        """
        DatasetMeasure.__init__(self, **(kwargs))

        self.__combiner = combiner

    def __repr__(self, prefixes=None):
        if prefixes is None: prefixes = []
        if self.__combiner != SecondAxisSumOfAbs:
            prefixes.append("combiner=%s" % self.__combiner)
        return \
            super(FeaturewiseDatasetMeasure, self).__repr__(prefixes=prefixes)


    def _call(self, dataset):
        """Computes a per-feature-measure on a given `Dataset`.

        Behaves like a `DatasetMeasure`, but computes and returns a 1d ndarray
        with one value per feature.
        """
        raise NotImplementedError


    def _postcall(self, dataset, result):
        """Adjusts per-feature-measure for computed `result`


        TODO: overlaps in what it does heavily with
         CombinedSensitivityAnalyzer, thus this one might make use of
         CombinedSensitivityAnalyzer yoh thinks, and here
         base_sensitivities doesn't sound appropriate.
         MH: There is indeed some overlap, but also significant differences.
             This one operates on a single sensana and combines over second
             axis, CombinedFeaturewiseDatasetMeasure uses first axis.
             Additionally, 'Sensitivity' base class is
             FeaturewiseDatasetMeasures which would have to be changed to
             CombinedFeaturewiseDatasetMeasure to deal with stuff like
             SMLRWeights that return multiple sensitivity values by default.
             Not sure if unification of both (and/or removal of functionality
             here does not lead to an overall more complicated situation,
             without any real gain -- after all this one works ;-)
        """
        rsshape = result.squeeze().shape
        if len(result.squeeze().shape)>1:
            n_base = result.shape[1]
            """Number of base sensitivities"""
            if self.states.isEnabled('base_sensitivities'):
                b_sensitivities = []
                if not self.states.isKnown('biases'):
                    biases = None
                else:
                    biases = self.biases
                    if len(self.biases) != n_base:
                        raise ValueError, \
                          "Number of biases %d is " % len(self.biases) \
                          + "different from number of base sensitivities" \
                          + "%d" % n_base
                for i in xrange(n_base):
                    if not biases is None:
                        bias = biases[i]
                    else:
                        bias = None
                    b_sensitivities = StaticDatasetMeasure(
                        measure = result[:,i],
                        bias = bias)
                self.base_sensitivities = b_sensitivities

            # After we stored each sensitivity separately,
            # we can apply combiner
            if self.__combiner is not None:
                result = self.__combiner(result)
        else:
            # remove bogus dimensions
            # XXX we might need to come up with smth better. May be some naive combiner? :-)
            result = result.squeeze()

        # call base class postcall
        result = DatasetMeasure._postcall(self, dataset, result)

        return result



class StaticDatasetMeasure(DatasetMeasure):
    """A static (assigned) sensitivity measure.

    Since implementation is generic it might be per feature or
    per whole dataset
    """

    def __init__(self, measure=None, bias=None, *args, **kwargs):
        """Initialize.

        :Parameters:
          measure
             actual sensitivity to be returned
          bias
             optionally available bias
        """
        DatasetMeasure.__init__(self, *(args), **(kwargs))
        if measure is None:
            raise ValueError, "Sensitivity measure has to be provided"
        self.__measure = measure
        self.__bias = bias

    def _call(self, dataset):
        """Returns assigned sensitivity
        """
        return self.__measure

    #XXX Might need to move into StateVariable?
    bias = property(fget=lambda self:self.__bias)



#
# Flavored implementations of FeaturewiseDatasetMeasures

class Sensitivity(FeaturewiseDatasetMeasure):

    _LEGAL_CLFS = []
    """If Sensitivity is classifier specific, classes of classifiers
    should be listed in the list
    """

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
        FeaturewiseDatasetMeasure.__init__(self, **kwargs)

        _LEGAL_CLFS = self._LEGAL_CLFS
        if len(_LEGAL_CLFS) > 0:
            found = False
            for clf_class in _LEGAL_CLFS:
                if isinstance(clf, clf_class):
                    found = True
                    break
            if not found:
                raise ValueError, \
                  "Classifier %s has to be of allowed class (%s), but is %s" \
                              % (clf, _LEGAL_CLFS, `type(clf)`)

        self.__clf = clf
        """Classifier used to computed sensitivity"""

        self._force_training = force_training
        """Either to force it to train"""

    def __repr__(self, prefixes=None):
        if prefixes is None: prefixes = []
        prefixes.append("clf=%s" % repr(self.clf))
        if not self._force_training:
            prefixes.append("force_training=%s" % self._force_training)
        return super(Sensitivity, self).__repr__(prefixes=prefixes)


    def __call__(self, dataset=None):
        """Train classifier on `dataset` and then compute actual sensitivity.

        If the classifier is already trained it is possible to extract the
        sensitivities without passing a dataset.
        """
        # local bindings
        clf = self.__clf
        if not clf.trained or self._force_training:
            if dataset is None:
                raise ValueError, \
                      "Training classifier to compute sensitivities requires " \
                      "a dataset."
            if __debug__:
                debug("SA", "Training classifier %s %s" %
                      (`clf`,
                       {False: "since it wasn't yet trained",
                        True:  "although it was trained previousely"}
                       [clf.trained]))
            clf.train(dataset)

        return FeaturewiseDatasetMeasure.__call__(self, dataset)


    def _setClassifier(self, clf):
        self.__clf = clf

    clf = property(fget=lambda self:self.__clf,
                   fset=_setClassifier)



class CombinedFeaturewiseDatasetMeasure(FeaturewiseDatasetMeasure):
    """Set sensitivity analyzers to be merged into a single output"""

    sensitivities = StateVariable(enabled=False,
        doc="Sensitivities produced by each classifier")

    def __init__(self, analyzers=None,
                 combiner=FirstAxisMean,
                 **kwargs):
        if analyzers == None:
            analyzers = []

        FeaturewiseDatasetMeasure.__init__(self, **kwargs)
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
        if __debug__:
            debug("SA",
                  "Returning combined using %s sensitivity across %d items" %
                  (`self.__combiner`, len(sensitivities)))

        if self.__combiner is not None:
            sensitivities = self.__combiner(sensitivities)
        return sensitivities


    def _setAnalyzers(self, analyzers):
        """Set the analyzers
        """
        self.__analyzers = analyzers
        """Analyzers to use"""

    analyzers = property(fget=lambda x:x.__analyzers,
                         fset=_setAnalyzers,
                         doc="Used analyzers")



class BoostedClassifierSensitivityAnalyzer(Sensitivity):
    """Set sensitivity analyzers to be merged into a single output"""

    def __init__(self,
                 clf,
                 analyzer=None,
                 combined_analyzer=None,
                 **kwargs):
        """Initialize Sensitivity Analyzer for `BoostedClassifier`
        """
        Sensitivity.__init__(self, clf, **kwargs)
        if combined_analyzer is None:
            # sanitarize kwargs
            kwargs.pop('force_training', None)
            combined_analyzer = CombinedFeaturewiseDatasetMeasure(**kwargs)
        self.__combined_analyzer = combined_analyzer
        """Combined analyzer to use"""

        self.__analyzer = None
        """Analyzer to use for basic classifiers within boosted classifier"""


    def _call(self, dataset):
        analyzers = []
        # create analyzers
        for clf in self.clf.clfs:
            if self.__analyzer is None:
                analyzer = clf.getSensitivityAnalyzer()
                if analyzer is None:
                    raise ValueError, \
                          "Wasn't able to figure basic analyzer for clf %s" % \
                          `clf`
                if __debug__:
                    debug("SA", "Selected analyzer %s for clf %s" % \
                          (`analyzer`, `clf`))
            else:
                # XXX shallow copy should be enough...
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


class ProxyClassifierSensitivityAnalyzer(Sensitivity):
    """Set sensitivity analyzer output just to pass through"""

    def __init__(self,
                 clf,
                 analyzer=None,
                 **kwargs):
        """Initialize Sensitivity Analyzer for `BoostedClassifier`
        """
        Sensitivity.__init__(self, clf, **kwargs)

        self.__analyzer = None
        """Analyzer to use for basic classifiers within boosted classifier"""


    def _call(self, dataset):
        if self.__analyzer is None:
            self.__analyzer = self.clf.clf.getSensitivityAnalyzer()
            if self.__analyzer is None:
                raise ValueError, \
                      "Wasn't able to figure basic analyzer for clf %s" % \
                      `self.clf.clf`
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
