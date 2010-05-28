# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Base classes for measures: algorithms that quantify properties of datasets.

Besides the `Measure` base class this module also provides the
(abstract) `FeaturewiseMeasure` class. The difference between a general
measure and the output of the `FeaturewiseMeasure` is that the latter
returns a 1d map (one value per feature in the dataset). In contrast there are
no restrictions on the returned value of `Measure` except for that it
has to be in some iterable container.

"""

__docformat__ = 'restructuredtext'

import numpy as np
import mvpa.support.copy as copy

from mvpa.base.learner import Learner
from mvpa.base.state import ConditionalAttribute
from mvpa.misc.args import group_kwargs
from mvpa.misc.attrmap import AttributeMap
from mvpa.misc.errorfx import mean_mismatch_error
from mvpa.base.types import asobjarray

from mvpa.base.dochelpers import enhanced_doc_string
from mvpa.base import externals, warning
from mvpa.clfs.stats import auto_null_dist
from mvpa.base.dataset import AttrDataset
from mvpa.datasets import Dataset, vstack
from mvpa.mappers.fx import BinaryFxNode
from mvpa.generators.splitters import Splitter

if __debug__:
    from mvpa.base import debug


class Measure(Learner):
    """A measure computed from a `Dataset`

    All dataset measures support arbitrary transformation of the measure
    after it has been computed. Transformation are done by processing the
    measure with a functor that is specified via the `transformer` keyword
    argument of the constructor. Upon request, the raw measure (before
    transformations are applied) is stored in the `raw_results` conditional attribute.

    Additionally all dataset measures support the estimation of the
    probabilit(y,ies) of a measure under some distribution. Typically this will
    be the NULL distribution (no signal), that can be estimated with
    permutation tests. If a distribution estimator instance is passed to the
    `null_dist` keyword argument of the constructor the respective
    probabilities are automatically computed and stored in the `null_prob`
    conditional attribute.

    Notes
    -----
    For developers: All subclasses shall get all necessary parameters via
    their constructor, so it is possible to get the same type of measure for
    multiple datasets by passing them to the __call__() method successively.

    """

    raw_results = ConditionalAttribute(enabled=False,
        doc="Computed results before applying any " +
            "transformation algorithm")
    null_prob = ConditionalAttribute(enabled=True)
    """Stores the probability of a measure under the NULL hypothesis"""
    null_t = ConditionalAttribute(enabled=False)
    """Stores the t-score corresponding to null_prob under assumption
    of Normal distribution"""

    def __init__(self, postproc=None, null_dist=None, **kwargs):
        """Does nothing special.

        Parameters
        ----------
        postproc : Mapper instance
          Mapper to perform post-processing of results. This mapper is applied
          in `__call__()` to perform a final processing step on the to be
          returned dataset measure. If None, nothing is done.
        null_dist : instance of distribution estimator
          The estimated distribution is used to assign a probability for a
          certain value of the computed measure.
        """
        Learner.__init__(self, **kwargs)

        self.__postproc = postproc
        """Functor to be called in return statement of all subclass __call__()
        methods."""
        null_dist_ = auto_null_dist(null_dist)
        if __debug__:
            debug('SA', 'Assigning null_dist %s whenever original given was %s'
                  % (null_dist_, null_dist))
        self.__null_dist = null_dist_


    __doc__ = enhanced_doc_string('Measure', locals(),
                                  Learner)


    def _postcall(self, dataset, result):
        """Some postprocessing on the result
        """
        self.ca.raw_results = result

        # post-processing
        if not self.__postproc is None:
            if __debug__:
                debug("SA_",
                      "Applying post-processing node %s" % self.__postproc)
            result = self.__postproc(result)

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

            if self.ca.is_enabled('null_t'):
                # get probability under NULL hyp, but also request
                # either it belong to the right tail
                null_prob, null_right_tail = \
                           self.__null_dist.p(result, return_tails=True)
                self.ca.null_prob = null_prob

                externals.exists('scipy', raise_=True)
                from scipy.stats import norm

                # TODO: following logic should appear in NullDist,
                #       not here
                tail = self.null_dist.tail
                if tail == 'left':
                    acdf = np.abs(null_prob)
                elif tail == 'right':
                    acdf = 1.0 - np.abs(null_prob)
                elif tail in ['any', 'both']:
                    acdf = 1.0 - np.clip(np.abs(null_prob), 0, 0.5)
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
                null_t = norm.ppf(np.clip(acdf, clip, 1.0 - clip))
                # assure that we deal with arrays:
                null_t = np.array(null_t, ndmin=1, copy=False)
                null_t[~null_right_tail] *= -1.0 # revert sign for negatives
                self.ca.null_t = null_t          # store
            else:
                # get probability of result under NULL hypothesis if available
                # and don't request tail information
                self.ca.null_prob = self.__null_dist.p(result)

        return result


    def __repr__(self, prefixes=None):
        """String representation of a `Measure`

        Includes only arguments which differ from default ones
        """
        if prefixes is None:
            prefixes = []
        prefixes = prefixes[:]
        if self.__postproc is not None:
            prefixes.append("postproc=%s" % self.__postproc)
        if self.__null_dist is not None:
            prefixes.append("null_dist=%s" % self.__null_dist)
        return super(Measure, self).__repr__(prefixes=prefixes)


    @property
    def null_dist(self):
        """Return Null Distribution estimator"""
        return self.__null_dist


    @property
    def postproc(self):
        """Return mapper"""
        return self.__postproc



class RepeatedMeasure(Measure):
    """Repeatedly run a measure on generated dataset.

    A measure is ran multiple times on datasets yielded by a custom generator.
    Results of all measure runs are stacked and returned as a dataset upon call.
    """

    repetition_results = ConditionalAttribute(enabled=False, doc=
       """Store individual result datasets for each repetition""")
    stats = ConditionalAttribute(enabled=False, doc=
       """Summary statistics about the node performance across all repetitions
       """)
    datasets = ConditionalAttribute(enabled=False, doc=
       """Store generated datasets for all repetitions. Can be memory expensive
       """)

    def __init__(self,
                 node,
                 generator,
                 **kwargs):
        """
        Parameters
        ----------
        node : Node
          Node or Measure implementing the procedure that is supposed to be run
          multiple times.
        generator : Node
          Generator to yield a dataset for each measure run. The number of
          datasets returned by the node determines the number of runs.
        """
        Measure.__init__(self, **kwargs)

        self.__node = node
        self.__generator = generator


    def _call(self, ds):
        # local binding
        generator = self.__generator
        node = self.__node
        ca = self.ca

        if self.ca.is_enabled("stats") and (not node.ca.has_key("stats") or
                                            not node.ca.is_enabled("stats")):
            raise ValueError("'stats' conditional attribute was enabled, but "
                             "the assigned node either doesn't support it, "
                             "or it is disabled")
        # precharge conditional attributes
        ca.datasets = []

        # run the node an all generated datasets
        results = []
        for sds in generator.generate(ds):
            if ca.is_enabled("datasets"):
                # store dataset in ca
                ca.datasets.append(sds)
            # run the beast
            result = node(sds)
            results.append(result)

            # subclass postprocessing
            self._repetition_postcall(sds, node, result)

            if ca.is_enabled("stats"):
                if not ca.is_set('stats'):
                    # create empty stats container of matching type
                    ca.stats = node.ca['stats'].value.__class__()
                # harvest summary stats
                ca['stats'].value.__iadd__(node.ca['stats'].value)

        # charge condition attribute
        self.ca.repetition_results = results

        # stack all results into a single Dataset
        results = vstack(results)
        # no need to store the raw results, since the Measure class will
        # automatically store them in a CA
        return results


    def _repetition_postcall(self, ds, node, result):
        """Post-processing handler for each repetition.

        Maybe overwritten in subclasses to harvest additional data.

        Parameters
        ----------
        ds : Dataset
          Input dataset for the node for this repetition
        node : Node
          Node after having processed the input dataset
        result : Dataset
          Output dataset of the node for this repetition.
        """
        pass



class CrossValidation(RepeatedMeasure):
    """Cross-validate a learner's transfer on datasets.

    A generator is used to resample a dataset into multiple instances (e.g.
    sets of dataset partitions for leave-one-out folding). For each dataset
    instance a transfer measure is computed by splitting the dataset into
    two parts (defined by the dataset generators output space) and train a
    custom learner on the first part and run it on the next. An arbitray error
    function can by used to determine the learner's error when prediction the
    dataset part that has been unseen during training.
    """

    training_stats = ConditionalAttribute(enabled=False, doc=
       """Summary statistics about the training status of the learner
       across all cross-validation fold.""")

    # TODO move conditional attributes from CVTE into this guy
    def __init__(self, learner, generator, errorfx=mean_mismatch_error,
                 space='targets', **kwargs):
        """
        Parameters
        ----------
        learner : Learner
          Any trainable node that shall be run on the dataset folds.
        generator : Node
          Generator used to resample the input dataset into multiple instances
          (i.e. partitioning it). The number of datasets yielded by this
          generator determines the number of cross-validation folds.
        errorfx : callable
          Custom implementation of an error function. The callable needs to
          accept two arguments (1. predicted values, 2. target values).
        space : str
          Target space of the learner, i.e. the sample attribute it will be
          trained on and tries to predict.
        """
        # compile the appropriate repeated measure to do cross-validation from
        # pieces
        if not errorfx is None:
            # error node -- postproc of transfer measure
            enode = BinaryFxNode(mean_mismatch_error, space)
        else:
            enode = Node

        # enforce learner's space
        # XXX maybe not in all cases?
        learner.set_space(space)

        # transfer measure to wrap the learner
        # splitter used the output space of the generator to know what to split
        tm = TransferMeasure(learner, Splitter(generator.get_space()),
                postproc=enode)

        # and finally the repeated measure to perform the x-val
        RepeatedMeasure.__init__(self, tm, generator, **kwargs)

        for ca in ['stats', 'training_stats']:
            if self.ca.is_enabled(ca):
                # enforce ca if requested
                tm.ca.enable(ca)
        if self.ca.is_enabled('training_stats'):
            # also enable training stats in the learner
            # TODO this needs to become 'training_stats' whenever the
            # classifiers are ready
            learner.ca.enable('training_confusion')


    def _repetition_postcall(self, ds, node, result):
        # local binding
        ca = self.ca
        if ca.is_enabled("training_stats"):
            if not ca.is_set('training_stats'):
                # create empty stats container of matching type
                ca.training_stats = node.ca['training_stats'].value.__class__()
            # harvest summary stats
            ca['training_stats'].value.__iadd__(node.ca['training_stats'].value)




class TransferMeasure(Measure):
    """Train and run a measure on two different parts of a dataset.

    Upon calling a TransferMeasure instance with a dataset the input dataset
    is passed to a `Splitter` to will generate dataset subsets. The first
    generated dataset is used to train an arbitray embedded `Measure. Once
    trained, the measure is then called with the second generated dataset
    and the result is returned.
    """

    stats = ConditionalAttribute(enabled=False, doc=
       """Optional summary statistics about the transfer performance""")
    training_stats = ConditionalAttribute(enabled=False, doc=
       """Summary statistics about the training status of the learner""")

    def __init__(self, measure, splitter, **kwargs):
        """
        Parameters
        ----------
        measure: Measure
          This measure instance is trained on the first dataset and called with
          the second.
        splitter: Splitter
          This splitter instance has to generate at least two dataset splits
          when called with the input dataset. The first split is used to train
          the measure, the second split is used to run the trained measure.
        """
        Measure.__init__(self, **kwargs)
        self.__measure = measure
        self.__splitter = splitter


    def _call(self, ds):
        # local binding
        measure = self.__measure
        splitter = self.__splitter
        ca = self.ca

        # activate the dataset splitter
        dsgen = splitter.generate(ds)
        # train on first
        measure.train(dsgen.next())

        # TODO get training confusion/stats

        # run with second
        res = measure(dsgen.next())

        # compute measure stats
        if ca.is_enabled('stats'):
            if not hasattr(measure, '__summary_class__'):
                raise ValueError('%s has no __summary_class__ attribute -- '
                                 'necessary for computing transfer stats'
                                 % measure)
            stats = measure.__summary_class__(
                # hmm, might be unsupervised, i.e no targets...
                targets=res.sa[measure.get_space()].value,
                predictions=res.samples.squeeze(),
                estimates = measure.ca.get('estimates', None))
            ca.stats = stats
        if ca.is_enabled('training_stats'):
            ca.training_stats = measure.ca.training_confusion

        return res



class FeaturewiseMeasure(Measure):
    """A per-feature-measure computed from a `Dataset` (base class).

    Should behave like a Measure.
    """

    # MH: why isn't this piece in the Sensitivity class?
    base_sensitivities = ConditionalAttribute(enabled=False,
        doc="Stores basic sensitivities if the sensitivity " +
            "relies on combining multiple ones")

    def __init__(self, **kwargs):
        Measure.__init__(self, **kwargs)

    def __repr__(self, prefixes=None):
        if prefixes is None:
            prefixes = []
        return \
            super(FeaturewiseMeasure, self).__repr__(prefixes=prefixes)


    def _postcall(self, dataset, result):
        """Adjusts per-feature-measure for computed `result`


        TODO: overlaps in what it does heavily with
         CombinedSensitivityAnalyzer, thus this one might make use of
         CombinedSensitivityAnalyzer yoh thinks, and here
         base_sensitivities doesn't sound appropriate.
         MH: There is indeed some overlap, but also significant differences.
             This one operates on a single sensana and combines over second
             axis, CombinedFeaturewiseMeasure uses first axis.
             Additionally, 'Sensitivity' base class is
             FeaturewiseMeasures which would have to be changed to
             CombinedFeaturewiseMeasure to deal with stuff like
             SMLRWeights that return multiple sensitivity values by default.
             Not sure if unification of both (and/or removal of functionality
             here does not lead to an overall more complicated situation,
             without any real gain -- after all this one works ;-)
        """
        # This method get the 'result' either as a 1D array, or as a Dataset
        # everything else is illegal
        if __debug__ \
               and not isinstance(result, AttrDataset) \
               and not len(result.shape) == 1:
            raise RuntimeError("FeaturewiseMeasures have to return "
                               "their results as 1D array, or as a Dataset "
                               "(error made by: '%s')." % repr(self))

        if len(result.shape) > 1:
            n_base = len(result)
            """Number of base sensitivities"""
            if self.ca.is_enabled('base_sensitivities'):
                b_sensitivities = []
                if not self.ca.has_key('biases'):
                    biases = None
                else:
                    biases = self.ca.biases
                    if len(self.ca.biases) != n_base:
                        warning("Number of biases %d differs from number "
                                "of base sensitivities %d which could happen "
                                "when measure is collided across labels."
                                % (len(self.ca.biases), n_base))
                for i in xrange(n_base):
                    if not biases is None:
                        if n_base > 1 and len(biases) == 1:
                            # The same bias for all bases
                            bias = biases[0]
                        else:
                            bias = biases[i]
                    else:
                        bias = None
                    b_sensitivities = StaticMeasure(
                        measure = result[i],
                        bias = bias)
                self.ca.base_sensitivities = b_sensitivities

        # XXX Remove when "sensitivity-return-dataset" transition is done
        if __debug__ \
           and not isinstance(result, AttrDataset) and not len(result.shape) == 1:
            warning("FeaturewiseMeasures-related post-processing "
                    "of '%s' doesn't return a Dataset, or 1D-array."
                    % self.__class__.__name__)

        # call base class postcall
        result = Measure._postcall(self, dataset, result)

        return result



class StaticMeasure(Measure):
    """A static (assigned) sensitivity measure.

    Since implementation is generic it might be per feature or
    per whole dataset
    """

    def __init__(self, measure=None, bias=None, *args, **kwargs):
        """Initialize.

        Parameters
        ----------
        measure
           actual sensitivity to be returned
        bias
           optionally available bias
        """
        Measure.__init__(self, *args, **kwargs)
        if measure is None:
            raise ValueError, "Sensitivity measure has to be provided"
        self.__measure = measure
        self.__bias = bias

    def _call(self, dataset):
        """Returns assigned sensitivity
        """
        return self.__measure

    #XXX Might need to move into ConditionalAttribute?
    bias = property(fget=lambda self:self.__bias)



#
# Flavored implementations of FeaturewiseMeasures

class Sensitivity(FeaturewiseMeasure):
    """Sensitivities of features for a given Classifier.

    """

    _LEGAL_CLFS = []
    """If Sensitivity is classifier specific, classes of classifiers
    should be listed in the list
    """

    def __init__(self, clf, force_training=True, **kwargs):
        """Initialize the analyzer with the classifier it shall use.

        Parameters
        ----------
        clf : `Classifier`
          classifier to use.
        force_training : bool
          if classifier was already trained -- do not retrain
        """

        """Does nothing special."""
        FeaturewiseMeasure.__init__(self, **kwargs)

        _LEGAL_CLFS = self._LEGAL_CLFS
        if len(_LEGAL_CLFS) > 0:
            found = False
            for clf_class in _LEGAL_CLFS:
                if isinstance(clf, clf_class):
                    found = True
                    break
            if not found:
                raise ValueError, \
                  "Classifier %s has to be of allowed class (%s), but is %r" \
                  % (clf, _LEGAL_CLFS, type(clf))

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
                      (repr(clf),
                       {False: "since it wasn't yet trained",
                        True:  "although it was trained previously"}
                       [clf.trained]))
            clf.train(dataset)

        return FeaturewiseMeasure.__call__(self, dataset)


    def _set_classifier(self, clf):
        self.__clf = clf


    def untrain(self):
        """Untrain corresponding classifier for Sensitivity
        """
        if self.__clf is not None:
            self.__clf.untrain()

    @property
    def feature_ids(self):
        """Return feature_ids used by the underlying classifier
        """
        return self.__clf._get_feature_ids()


    clf = property(fget=lambda self:self.__clf,
                   fset=_set_classifier)



class CombinedFeaturewiseMeasure(FeaturewiseMeasure):
    """Set sensitivity analyzers to be merged into a single output"""

    sensitivities = ConditionalAttribute(enabled=False,
        doc="Sensitivities produced by each analyzer")

    # XXX think again about combiners... now we have it in here and as
    #     well as in the parent -- FeaturewiseMeasure
    # YYY because we don't use parent's _call. Needs RF
    def __init__(self, analyzers=None,  # XXX should become actually 'measures'
                 sa_attr='combinations',
                 **kwargs):
        """Initialize CombinedFeaturewiseMeasure

        Parameters
        ----------
        analyzers : list or None
          List of analyzers to be used. There is no logic to populate
          such a list in __call__, so it must be either provided to
          the constructor or assigned to .analyzers prior calling
        sa_attr : str
          Name of the sa to be populated with the indexes of combinations
        """
        if analyzers is None:
            analyzers = []
        self._sa_attr = sa_attr
        FeaturewiseMeasure.__init__(self, **kwargs)
        self.__analyzers = analyzers
        """List of analyzers to use"""


    def _call(self, dataset):
        sensitivities = []
        for ind, analyzer in enumerate(self.__analyzers):
            if __debug__:
                debug("SA", "Computing sensitivity for SA#%d:%s" %
                      (ind, analyzer))
            sensitivity = analyzer(dataset)
            sensitivities.append(sensitivity)

        if __debug__:
            debug("SA",
                  "Returning %d sensitivities from %s" %
                  (len(sensitivities), self.__class__.__name__))

        sa_attr = self._sa_attr
        if isinstance(sensitivities[0], AttrDataset):
            smerged = None
            for i, s in enumerate(sensitivities):
                s.sa[sa_attr] = np.repeat(i, len(s))
                if smerged is None:
                    smerged = s
                else:
                    smerged.append(s)
            sensitivities = smerged
        else:
            sensitivities = \
                Dataset(sensitivities,
                        sa={sa_attr: np.arange(len(sensitivities))})

        self.ca.sensitivities = sensitivities

        return sensitivities


    def untrain(self):
        """Untrain CombinedFDM
        """
        if self.__analyzers is not None:
            for anal in self.__analyzers:
                anal.untrain()

    ##REF: Name was automagically refactored
    def _set_analyzers(self, analyzers):
        """Set the analyzers
        """
        self.__analyzers = analyzers
        """Analyzers to use"""

    analyzers = property(fget=lambda x:x.__analyzers,
                         fset=_set_analyzers,
                         doc="Used analyzers")


# XXX Why did we come to name everything analyzer? inputs of regular
#     things like CombinedFeaturewiseMeasure can be simple
#     measures....

class SplitFeaturewiseMeasure(FeaturewiseMeasure):
    """Compute measures across splits for a specific analyzer"""

    # XXX This beast is created based on code of
    #     CombinedFeaturewiseMeasure, thus another reason to refactor

    sensitivities = ConditionalAttribute(enabled=False,
        doc="Sensitivities produced for each split")

    splits = ConditionalAttribute(enabled=False, doc=
       """Store the actual splits of the data. Can be memory expensive""")

    def __init__(self, splitter, analyzer,
                 insplit_index=0, **kwargs):
        """Initialize SplitFeaturewiseMeasure

        Parameters
        ----------
        splitter : Splitter
          Splitter to use to split the dataset
        analyzer : Measure
          Measure to be used. Could be analyzer as well (XXX)
        insplit_index : int
          splitter generates tuples of dataset on each iteration
          (usually 0th for training, 1st for testing).
          On what split index in that tuple to operate.
        """

        # XXX might want to extend insplit_index to handle 'all', so we store
        #     sensitivities for all parts of the splits... not sure if it is needed

        # XXX We really think through whole transformer/combiners pipelining

        # Here we provide mapper None since the postprocessing should be done
        # at the toplevel and just once
        FeaturewiseMeasure.__init__(self, postproc=None, **kwargs)

        self.__analyzer = analyzer
        """Analyzer to use per split"""

        self.__splitter = splitter
        """Splitter to be used on the dataset"""

        self.__insplit_index = insplit_index


    def untrain(self):
        """Untrain SplitFeaturewiseMeasure
        """
        if self.__analyzer is not None:
            self.__analyzer.untrain()


    def _call(self, dataset):
        # local bindings
        analyzer = self.__analyzer
        insplit_index = self.__insplit_index

        sensitivities = []
        self.ca.splits = splits = []
        store_splits = self.ca.is_enabled("splits")

        for ind, split in enumerate(self.__splitter(dataset)):
            ds = split[insplit_index]
            if __debug__ and "SA" in debug.active:
                debug("SA", "Computing sensitivity for split %d on "
                      "dataset %s using %s" % (ind, ds, analyzer))
            sensitivity = analyzer(ds)
            sensitivities.append(sensitivity)
            if store_splits:
                splits.append(split)

        result = vstack(sensitivities)
        result.sa['splits'] = np.concatenate([[i] * len(s)
                                for i, s in enumerate(sensitivities)])
        self.ca.sensitivities = sensitivities
        return result


class BoostedClassifierSensitivityAnalyzer(Sensitivity):
    """Set sensitivity analyzers to be merged into a single output"""


    # XXX we might like to pass parameters also for combined_analyzer
    @group_kwargs(prefixes=['slave_'], assign=True)
    def __init__(self,
                 clf,
                 analyzer=None,
                 combined_analyzer=None,
                 sa_attr='lrn_index',
                 slave_kwargs={},
                 **kwargs):
        """Initialize Sensitivity Analyzer for `BoostedClassifier`

        Parameters
        ----------
        clf : `BoostedClassifier`
          Classifier to be used
        analyzer : analyzer
          Is used to populate combined_analyzer
        sa_attr : str
          Name of the sa to be populated with the indexes of learners
          (passed to CombinedFeaturewiseMeasure is None is
          given in `combined_analyzer`)
        slave_*
          Arguments to pass to created analyzer if analyzer is None
        """
        Sensitivity.__init__(self, clf, **kwargs)
        if combined_analyzer is None:
            # sanitarize kwargs
            kwargs.pop('force_training', None)
            combined_analyzer = CombinedFeaturewiseMeasure(sa_attr=sa_attr,
                                                                  **kwargs)
        self.__combined_analyzer = combined_analyzer
        """Combined analyzer to use"""

        if analyzer is not None and len(self._slave_kwargs):
            raise ValueError, \
                  "Provide either analyzer of slave_* arguments, not both"
        self.__analyzer = analyzer
        """Analyzer to use for basic classifiers within boosted classifier"""


    def untrain(self):
        """Untrain BoostedClassifierSensitivityAnalyzer
        """
        if self.__analyzer is not None:
            self.__analyzer.untrain()
        self.__combined_analyzer.untrain()


    def _call(self, dataset):
        analyzers = []
        # create analyzers
        for clf in self.clf.clfs:
            if self.__analyzer is None:
                analyzer = clf.get_sensitivity_analyzer(**(self._slave_kwargs))
                if analyzer is None:
                    raise ValueError, \
                          "Wasn't able to figure basic analyzer for clf %r" % \
                          (clf,)
                if __debug__:
                    debug("SA", "Selected analyzer %r for clf %r" % \
                          (analyzer, clf))
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

    clf_sensitivities = ConditionalAttribute(enabled=False,
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


    def untrain(self):
        super(ProxyClassifierSensitivityAnalyzer, self).untrain()
        if self.__analyzer is not None:
            self.__analyzer.untrain()


    def _call(self, dataset):
        # OPT: local bindings
        clfclf = self.clf.clf
        analyzer = self.__analyzer

        if analyzer is None:
            analyzer = clfclf.get_sensitivity_analyzer(
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
        self.ca.clf_sensitivities = result

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


class BinaryClassifierSensitivityAnalyzer(ProxyClassifierSensitivityAnalyzer):
    """Set sensitivity analyzer output to have proper labels"""

    def _call(self, dataset):
        sens = super(self.__class__, self)._call(dataset)
        clf = self.clf
        targets_attr = clf.get_space()
        if targets_attr in sens.sa:
            # if labels are present -- transform them into meaningful tuples
            # (or not if just a single beast)
            am = AttributeMap(dict([(l, -1) for l in clf.neglabels] +
                                   [(l, +1) for l in clf.poslabels]))

            # XXX here we still can get a sensitivity per each label
            # (e.g. with SMLR as the slave clf), so I guess we should
            # tune up Multiclass...Analyzer to add an additional sa
            # And here we might need to check if asobjarray call is necessary
            # and should be actually done
            #asobjarray(
            sens.sa[targets_attr] = \
                am.to_literal(sens.sa[targets_attr].value, recurse=True)
        return sens


class RegressionAsClassifierSensitivityAnalyzer(ProxyClassifierSensitivityAnalyzer):
    """Set sensitivity analyzer output to have proper labels"""

    def _call(self, dataset):
        sens = super(RegressionAsClassifierSensitivityAnalyzer,
                     self)._call(dataset)
        # We can have only a single sensitivity out of regression
        assert(sens.shape[0] == 1)
        clf = self.clf
        targets_attr = clf.get_space()
        if targets_attr not in sens.sa:
            # We just assign a tuple of all labels sorted
            labels = tuple(sorted(clf._trained_attrmap.values()))
            if len(clf._trained_attrmap):
                labels = clf._trained_attrmap.to_literal(labels, recurse=True)
            sens.sa[targets_attr] = asobjarray([labels])
        return sens


class FeatureSelectionClassifierSensitivityAnalyzer(
    ProxyClassifierSensitivityAnalyzer):
    """Set sensitivity analyzer output be reverse mapped using mapper of the
    slave classifier"""

    def _call(self, dataset):
        sens = super(FeatureSelectionClassifierSensitivityAnalyzer,
                     self)._call(dataset)
        # `sens` is either 1D array, or Dataset
        if isinstance(sens, AttrDataset):
            return self.clf.maskclf.mapper.reverse(sens)
        else:
            return self.clf.maskclf.mapper.reverse1(sens)
