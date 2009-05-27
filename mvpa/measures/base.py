# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
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

import numpy as N
import mvpa.support.copy as copy

from mvpa.misc.state import StateVariable, ClassWithCollections
from mvpa.misc.args import group_kwargs
from mvpa.misc.transformers import FirstAxisMean, SecondAxisSumOfAbs
from mvpa.base.dochelpers import enhancedDocString
from mvpa.base import externals
from mvpa.clfs.stats import autoNullDist

if __debug__:
    from mvpa.base import debug


class DatasetMeasure(ClassWithCollections):
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

    .. note::
      For developers: All subclasses shall get all necessary parameters via
      their constructor, so it is possible to get the same type of measure for
      multiple datasets by passing them to the __call__() method successively.
    """

    raw_result = StateVariable(enabled=False,
        doc="Computed results before applying any " +
            "transformation algorithm")
    null_prob = StateVariable(enabled=True)
    """Stores the probability of a measure under the NULL hypothesis"""
    null_t = StateVariable(enabled=False)
    """Stores the t-score corresponding to null_prob under assumption
    of Normal distribution"""

    def __init__(self, transformer=None, null_dist=None, **kwargs):
        """Does nothing special.

        :Parameters:
          transformer: Functor
            This functor is called in `__call__()` to perform a final
            processing step on the to be returned dataset measure. If None,
            nothing is called
          null_dist: instance of distribution estimator
            The estimated distribution is used to assign a probability for a
            certain value of the computed measure.
        """
        ClassWithCollections.__init__(self, **kwargs)

        self.__transformer = transformer
        """Functor to be called in return statement of all subclass __call__()
        methods."""
        null_dist_ = autoNullDist(null_dist)
        if __debug__:
            debug('SA', 'Assigning null_dist %s whenever original given was %s'
                  % (null_dist_, null_dist))
        self.__null_dist = null_dist_


    __doc__ = enhancedDocString('DatasetMeasure', locals(), ClassWithCollections)


    def __call__(self, dataset):
        """Compute measure on a given `Dataset`.

        Each implementation has to handle a single arguments: the source
        dataset.

        Returns the computed measure in some iterable (list-like)
        container applying transformer if such is defined
        """
        result = self._call(dataset)
        result = self._postcall(dataset, result)
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
        self.raw_result = result
        if not self.__transformer is None:
            if __debug__:
                debug("SA_", "Applying transformer %s" % self.__transformer)
            result = self.__transformer(result)

        # estimate the NULL distribution when functor is given
        if not self.__null_dist is None:
            if __debug__:
                debug("SA_", "Estimating NULL distribution using %s"
                      % self.__null_dist)

            # we need a matching datameasure instance, but we have to disable
            # the estimation of the null distribution in that child to prevent
            # infinite looping.
            measure = copy.copy(self)
            measure.__null_dist = None
            self.__null_dist.fit(measure, dataset)

            if self.states.isEnabled('null_t'):
                # get probability under NULL hyp, but also request
                # either it belong to the right tail
                null_prob, null_right_tail = \
                           self.__null_dist.p(result, return_tails=True)
                self.null_prob = null_prob

                externals.exists('scipy', raiseException=True)
                from scipy.stats import norm

                # TODO: following logic should appear in NullDist,
                #       not here
                tail = self.null_dist.tail
                if tail == 'left':
                    acdf = N.abs(null_prob)
                elif tail == 'right':
                    acdf = 1.0 - N.abs(null_prob)
                elif tail in ['any', 'both']:
                    acdf = 1.0 - N.clip(N.abs(null_prob), 0, 0.5)
                else:
                    raise RuntimeError, 'Unhandled tail %s' % tail
                # We need to clip to avoid non-informative inf's ;-)
                # that happens due to lack of precision in mantissa
                # which is 11 bits in double. We could clip values
                # around 0 at as low as 1e-100 (correspond to z~=21),
                # but for consistency lets clip at 1e-16 which leads
                # to distinguishable value around p=1 and max z=8.2.
                # Should be sufficient range of z-values ;-)
                clip = 1e-16
                null_t = norm.ppf(N.clip(acdf, clip, 1.0 - clip))
                null_t[~null_right_tail] *= -1.0 # revert sign for negatives
                self.null_t = null_t                 # store
            else:
                # get probability of result under NULL hypothesis if available
                # and don't request tail information
                self.null_prob = self.__null_dist.p(result)

        return result


    def __repr__(self, prefixes=[]):
        """String representation of DatasetMeasure

        Includes only arguments which differ from default ones
        """
        prefixes = prefixes[:]
        if self.__transformer is not None:
            prefixes.append("transformer=%s" % self.__transformer)
        if self.__null_dist is not None:
            prefixes.append("null_dist=%s" % self.__null_dist)
        return super(DatasetMeasure, self).__repr__(prefixes=prefixes)


    @property
    def null_dist(self):
        """Return Null Distribution estimator"""
        return self.__null_dist

    @property
    def transformer(self):
        """Return transformer"""
        return self.__transformer


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
    # YYY in comparison to CombinedSensitivityAnalyzer here default
    #     value for combiner is worse than anywhere. From now on,
    #     default combiners should be provided "in place", ie
    #     in SMLR it makes sense to have SecondAxisMaxOfAbs,
    #     in SVM (pair-wise) only for not-binary should be
    #     SecondAxisSumOfAbs, though could be Max as well... uff
    #   YOH: started to do so, but still have issues... thus
    #        reverting back for now
    #   MH: Full ack -- voting for no default combiners!
    def __init__(self, combiner=SecondAxisSumOfAbs, **kwargs): # SecondAxisSumOfAbs
        """Initialize

        :Parameters:
          combiner : Functor
            The combiner is only applied if the computed featurewise dataset
            measure is more than one-dimensional. This is different from a
            `transformer`, which is always applied. By default, the sum of
            absolute values along the second axis is computed.
        """
        DatasetMeasure.__init__(self, **kwargs)

        self.__combiner = combiner

    def __repr__(self, prefixes=None):
        if prefixes is None:
            prefixes = []
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
        result_sq = result.squeeze()
        if len(result_sq.shape)>1:
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
            # XXX we might need to come up with smth better. May be some naive
            # combiner? :-)
            result = result_sq

        # call base class postcall
        result = DatasetMeasure._postcall(self, dataset, result)

        return result

    @property
    def combiner(self):
        """Return combiner"""
        return self.__combiner



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
        DatasetMeasure.__init__(self, *args, **kwargs)
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
          clf : :class:`Classifier`
            classifier to use.
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
        if prefixes is None:
            prefixes = []
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


    @property
    def feature_ids(self):
        """Return feature_ids used by the underlying classifier
        """
        return self.__clf._getFeatureIds()


    clf = property(fget=lambda self:self.__clf,
                   fset=_setClassifier)



class CombinedFeaturewiseDatasetMeasure(FeaturewiseDatasetMeasure):
    """Set sensitivity analyzers to be merged into a single output"""

    sensitivities = StateVariable(enabled=False,
        doc="Sensitivities produced by each analyzer")

    # XXX think again about combiners... now we have it in here and as
    #     well as in the parent -- FeaturewiseDatasetMeasure
    # YYY because we don't use parent's _call. Needs RF
    def __init__(self, analyzers=None,  # XXX should become actually 'measures'
                 combiner=None, #FirstAxisMean,
                 **kwargs):
        """Initialize CombinedFeaturewiseDatasetMeasure

        :Parameters:
          analyzers : list or None
            List of analyzers to be used. There is no logic to populate
            such a list in __call__, so it must be either provided to
            the constructor or assigned to .analyzers prior calling
        """
        if analyzers is None:
            analyzers = []

        FeaturewiseDatasetMeasure.__init__(self, **kwargs)
        self.__analyzers = analyzers
        """List of analyzers to use"""

        self.__combiner = combiner
        """Which functor to use to combine all sensitivities"""


    def _call(self, dataset):
        sensitivities = []
        for ind,analyzer in enumerate(self.__analyzers):
            if __debug__:
                debug("SA", "Computing sensitivity for SA#%d:%s" %
                      (ind, analyzer))
            sensitivity = analyzer(dataset)
            sensitivities.append(sensitivity)

        self.sensitivities = sensitivities
        if __debug__:
            debug("SA",
                  "Returning combined using %s sensitivity across %d items" %
                  (self.__combiner, len(sensitivities)))

        if self.__combiner is not None:
            sensitivities = self.__combiner(sensitivities)
        else:
            # assure that we have an ndarray on output
            sensitivities = N.asarray(sensitivities)
        return sensitivities


    def _setAnalyzers(self, analyzers):
        """Set the analyzers
        """
        self.__analyzers = analyzers
        """Analyzers to use"""

    analyzers = property(fget=lambda x:x.__analyzers,
                         fset=_setAnalyzers,
                         doc="Used analyzers")


# XXX Why did we come to name everything analyzer? inputs of regular
#     things like CombinedFeaturewiseDatasetMeasure can be simple
#     measures....

class SplitFeaturewiseDatasetMeasure(FeaturewiseDatasetMeasure):
    """Compute measures across splits for a specific analyzer"""

    # XXX This beast is created based on code of
    #     CombinedFeaturewiseDatasetMeasure, thus another reason to refactor

    sensitivities = StateVariable(enabled=False,
        doc="Sensitivities produced for each split")

    splits = StateVariable(enabled=False, doc=
       """Store the actual splits of the data. Can be memory expensive""")

    def __init__(self, splitter, analyzer,
                 insplit_index=0, combiner=None, **kwargs):
        """Initialize SplitFeaturewiseDatasetMeasure

        :Parameters:
          splitter : Splitter
            Splitter to use to split the dataset
          analyzer : DatasetMeasure
            Measure to be used. Could be analyzer as well (XXX)
          insplit_index : int
            splitter generates tuples of dataset on each iteration
            (usually 0th for training, 1st for testing).
            On what split index in that tuple to operate.
        """

        # XXX might want to extend insplit_index to handle 'all', so we store
        #     sensitivities for all parts of the splits... not sure if it is needed

        # XXX We really think through whole transformer/combiners pipelining

        # Here we provide combiner None since if needs to be combined
        # within each sensitivity, it better be done within analyzer
        FeaturewiseDatasetMeasure.__init__(self, combiner=None, **kwargs)

        self.__analyzer = analyzer
        """Analyzer to use per split"""

        self.__combiner = combiner
        """Which functor to use to combine all sensitivities"""

        self.__splitter = splitter
        """Splitter to be used on the dataset"""

        self.__insplit_index = insplit_index

    def _call(self, dataset):
        # local bindings
        analyzer = self.__analyzer
        insplit_index = self.__insplit_index

        sensitivities = []
        self.splits = splits = []
        store_splits = self.states.isEnabled("splits")

        for ind,split in enumerate(self.__splitter(dataset)):
            ds = split[insplit_index]
            if __debug__ and "SA" in debug.active:
                debug("SA", "Computing sensitivity for split %d on "
                      "dataset %s using %s" % (ind, ds, analyzer))
            sensitivity = analyzer(ds)
            sensitivities.append(sensitivity)
            if store_splits: splits.append(split)

        self.sensitivities = sensitivities
        if __debug__:
            debug("SA",
                  "Returning sensitivities combined using %s across %d items "
                  "generated by splitter %s" %
                  (self.__combiner, len(sensitivities), self.__splitter))

        if self.__combiner is not None:
            sensitivities = self.__combiner(sensitivities)
        else:
            # assure that we have an ndarray on output
            sensitivities = N.asarray(sensitivities)
        return sensitivities


class BoostedClassifierSensitivityAnalyzer(Sensitivity):
    """Set sensitivity analyzers to be merged into a single output"""


    # XXX we might like to pass parameters also for combined_analyzer
    @group_kwargs(prefixes=['slave_'], assign=True)
    def __init__(self,
                 clf,
                 analyzer=None,
                 combined_analyzer=None,
                 slave_kwargs={},
                 **kwargs):
        """Initialize Sensitivity Analyzer for `BoostedClassifier`

        :Parameters:
          clf : `BoostedClassifier`
            Classifier to be used
          analyzer : analyzer
            Is used to populate combined_analyzer 
          slave_*
            Arguments to pass to created analyzer if analyzer is None
        """
        Sensitivity.__init__(self, clf, **kwargs)
        if combined_analyzer is None:
            # sanitarize kwargs
            kwargs.pop('force_training', None)
            combined_analyzer = CombinedFeaturewiseDatasetMeasure(**kwargs)
        self.__combined_analyzer = combined_analyzer
        """Combined analyzer to use"""

        if analyzer is not None and len(self._slave_kwargs):
            raise ValueError, \
                  "Provide either analyzer of slave_* arguments, not both"
        self.__analyzer = analyzer
        """Analyzer to use for basic classifiers within boosted classifier"""


    def _call(self, dataset):
        analyzers = []
        # create analyzers
        for clf in self.clf.clfs:
            if self.__analyzer is None:
                analyzer = clf.getSensitivityAnalyzer(**(self._slave_kwargs))
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

        # XXX not sure if we don't want to call directly ._call(dataset) to avoid
        # double application of transformers/combiners, after all we are just
        # 'proxying' here to combined_analyzer...
        # YOH: decided -- lets call ._call
        return self.__combined_analyzer._call(dataset)

    combined_analyzer = property(fget=lambda x:x.__combined_analyzer)


class ProxyClassifierSensitivityAnalyzer(Sensitivity):
    """Set sensitivity analyzer output just to pass through"""

    clf_sensitivities = StateVariable(enabled=False,
        doc="Stores sensitivities of the proxied classifier")


    @group_kwargs(prefixes=['slave_'], assign=True)
    def __init__(self,
                 clf,
                 analyzer=None,
                 **kwargs):
        """Initialize Sensitivity Analyzer for `BoostedClassifier`
        """
        Sensitivity.__init__(self, clf, **kwargs)

        if analyzer is not None and len(self._slave_kwargs):
            raise ValueError, \
                  "Provide either analyzer of slave_* arguments, not both"

        self.__analyzer = analyzer
        """Analyzer to use for basic classifiers within boosted classifier"""


    def _call(self, dataset):
        # OPT: local bindings
        clfclf = self.clf.clf
        analyzer = self.__analyzer

        if analyzer is None:
            analyzer = clfclf.getSensitivityAnalyzer(
                **(self._slave_kwargs))
            if analyzer is None:
                raise ValueError, \
                      "Wasn't able to figure basic analyzer for clf %s" % \
                      `clfclf`
            if __debug__:
                debug("SA", "Selected analyzer %s for clf %s" % \
                      (analyzer, clfclf))
            # bind to the instance finally
            self.__analyzer = analyzer

        # TODO "remove" unnecessary things below on each call...
        # assign corresponding classifier
        analyzer.clf = clfclf

        # if clf was trained already - don't train again
        if clfclf.trained:
            analyzer._force_training = False

        result = analyzer._call(dataset)
        self.clf_sensitivities = result

        return result

    analyzer = property(fget=lambda x:x.__analyzer)


class MappedClassifierSensitivityAnalyzer(ProxyClassifierSensitivityAnalyzer):
    """Set sensitivity analyzer output be reverse mapped using mapper of the
    slave classifier"""

    def _call(self, dataset):
        sens = super(MappedClassifierSensitivityAnalyzer, self)._call(dataset)
        # So we have here the case that some sensitivities are given
        #  as nfeatures x nclasses, thus we need to take .T for the
        #  mapper and revert back afterwards
        # devguide's TODO lists this point to 'disguss'
        sens_mapped = self.clf.mapper.reverse(sens.T)
        return sens_mapped.T


class FeatureSelectionClassifierSensitivityAnalyzer(ProxyClassifierSensitivityAnalyzer):
    """Set sensitivity analyzer output be reverse mapped using mapper of the
    slave classifier"""

    def _call(self, dataset):
        sens = super(FeatureSelectionClassifierSensitivityAnalyzer, self)._call(dataset)
        # So we have here the case that some sensitivities are given
        #  as nfeatures x nclasses, thus we need to take .T for the
        #  mapper and revert back afterwards
        # devguide's TODO lists this point to 'disguss'
        sens_mapped = self.clf.maskclf.mapper.reverse(sens.T)
        return sens_mapped.T

