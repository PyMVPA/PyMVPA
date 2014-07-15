# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Plumbing for measures: algorithms that quantify properties of datasets.

Besides the `Measure` base class this module also provides the
(abstract) `FeaturewiseMeasure` class. The difference between a general
measure and the output of the `FeaturewiseMeasure` is that the latter
returns a 1d map (one value per feature in the dataset). In contrast there are
no restrictions on the returned value of `Measure` except for that it
has to be in some iterable container.

"""

__docformat__ = 'restructuredtext'

import numpy as np
import mvpa2.support.copy as copy

from mvpa2.base.node import Node
from mvpa2.base.learner import Learner
from mvpa2.base.state import ConditionalAttribute
from mvpa2.misc.args import group_kwargs
from mvpa2.misc.attrmap import AttributeMap
from mvpa2.misc.errorfx import mean_mismatch_error
from mvpa2.base.types import asobjarray

from mvpa2.base.dochelpers import enhanced_doc_string, _str, _repr_attrs
from mvpa2.base import externals, warning
from mvpa2.clfs.stats import auto_null_dist
from mvpa2.base.dataset import AttrDataset, vstack
from mvpa2.datasets import Dataset, vstack, hstack
from mvpa2.mappers.fx import BinaryFxNode
from mvpa2.generators.splitters import Splitter

if __debug__:
    from mvpa2.base import debug


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

    null_prob = ConditionalAttribute(enabled=True)
    """Stores the probability of a measure under the NULL hypothesis"""
    null_t = ConditionalAttribute(enabled=False)
    """Stores the t-score corresponding to null_prob under assumption
    of Normal distribution"""

    def __init__(self, null_dist=None, **kwargs):
        """
        Parameters
        ----------
        null_dist : instance of distribution estimator
          The estimated distribution is used to assign a probability for a
          certain value of the computed measure.
        """
        Learner.__init__(self, **kwargs)

        null_dist_ = auto_null_dist(null_dist)
        if __debug__:
            debug('SA', 'Assigning null_dist %s whenever original given was %s'
                  % (null_dist_, null_dist))
        self.__null_dist = null_dist_


    __doc__ = enhanced_doc_string('Measure', locals(),
                                  Learner)

    def __repr__(self, prefixes=[]):
        """String representation of a `Measure`

        Includes only arguments which differ from default ones
        """
        return super(Measure, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['null_dist']))


    def _precall(self, ds):
        # estimate the NULL distribution when functor is given
        if not self.__null_dist is None:
            if __debug__:
                debug("STAT", "Estimating NULL distribution using %s"
                      % self.__null_dist)

            # we need a matching measure instance, but we have to disable
            # the estimation of the null distribution in that child to prevent
            # infinite looping.
            measure = copy.copy(self)
            measure.__null_dist = None
            self.__null_dist.fit(measure, ds)


    def _postcall(self, dataset, result):
        """Some postprocessing on the result
        """
        if self.__null_dist is None:
            # do base-class postcall and be done
            result = super(Measure, self)._postcall(dataset, result)
        else:
            # don't do a full base-class postcall, only do the
            # postproc-application here, to gain result compatibility with the
            # fitted null distribution -- necessary to be able to use
            # a Node's 'pass_attr' to pick up ca.null_prob
            result = self._apply_postproc(dataset, result)

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
                    acdf = np.abs(null_prob.samples)
                elif tail == 'right':
                    acdf = 1.0 - np.abs(null_prob.samples)
                elif tail in ['any', 'both']:
                    acdf = 1.0 - np.clip(np.abs(null_prob.samples), 0, 0.5)
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
                null_t_ds = null_prob.copy(deep=False)
                null_t_ds.samples = null_t
                self.ca.null_t = null_t_ds          # store as a Dataset
            else:
                # get probability of result under NULL hypothesis if available
                # and don't request tail information
                self.ca.null_prob = self.__null_dist.p(result)
            # now do the second half of postcall and invoke pass_attr
            result = self._pass_attr(dataset, result)
        return result


    @property
    def null_dist(self):
        """Return Null Distribution estimator"""
        return self.__null_dist


class ProxyMeasure(Measure):
    """Wrapper to allow for alternative post-processing of a shared measure.

    This class is useful whenever a measure (or for example a trained
    classifier) shall be utilized in multiple nodes, but each node needs to
    perform its on post-processing of results. One can simply wrap the
    measure into this class and assign arbitrary post-processing nodes to the
    wrapper, instead of the measure itself.
    """

    def __init__(self, measure, skip_train=False, **kwargs):
        """
        Parameters
        ----------
        skip_train : bool, optional
          Flag whether the measure does not need to be really trained,
          since proxied measure is guaranteed to be trained appropriately
          prior to this call.  Use with caution
        """

        # by default auto train
        kwargs['auto_train'] = kwargs.get('auto_train', True)
        Measure.__init__(self, **kwargs)
        self.__measure = measure
        self.skip_train = skip_train

    def __repr__(self, prefixes=[]):
        """String representation of a `ProxyMeasure`
        """
        return super(ProxyMeasure, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['measure'])
            + _repr_attrs(self, ['skip_train'], default=False)
            )

    def _train(self, ds):
        if not self.skip_train:
            self.measure.train(ds)
        else:
            # only flag that it was trained
            self._set_trained()


    def _call(self, ds):
        return self.measure(ds)


    @property
    def measure(self):
        """Return proxied measure"""
        return self.__measure


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

    is_trained = True
    """Indicate that this measure is always trained."""

    def __init__(self,
                 node,
                 generator,
                 callback=None,
                 concat_as='samples',
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
        callback : functor
          Optional callback to extract information from inside the main loop of
          the measure. The callback is called with the input 'data', the 'node'
          instance that is evaluated repeatedly and the 'result' of a single
          evaluation -- passed as named arguments (see labels in quotes) for
          every iteration, directly after evaluating the node.
        concat_as : {'samples', 'features'}
          Along which axis to concatenate result dataset from all iterations.
          By default, results are 'vstacked' as multiple samples in the output
          dataset. Setting this argument to 'features' will change this to
          'hstacking' along the feature axis.
        """
        Measure.__init__(self, **kwargs)

        self._node = node
        self._generator = generator
        self._callback = callback
        self._concat_as = concat_as

    def __repr__(self, prefixes=[], exclude=[]):
        return super(RepeatedMeasure, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, [x for x in ['node', 'generator', 'callback']
                                 if not x in exclude])
            + _repr_attrs(self, ['concat_as'], default='samples')
            )


    def _call(self, ds):
        # local binding
        generator = self._generator
        node = self._node
        ca = self.ca
        space = self.get_space()
        concat_as = self._concat_as

        if self.ca.is_enabled("stats") and (not node.ca.has_key("stats") or
                                            not node.ca.is_enabled("stats")):
            warning("'stats' conditional attribute was enabled, but "
                    "the assigned node '%s' either doesn't support it, "
                    "or it is disabled" % node)
        # precharge conditional attributes
        ca.datasets = []

        # run the node an all generated datasets
        results = []
        for i, sds in enumerate(generator.generate(ds)):
            if __debug__:
                debug('REPM', "%d-th iteration of %s on %s",
                      (i, self, sds))
            if ca.is_enabled("datasets"):
                # store dataset in ca
                ca.datasets.append(sds)
            # run the beast
            result = node(sds)
            # callback
            if not self._callback is None:
                self._callback(data=sds, node=node, result=result)
            # subclass postprocessing
            result = self._repetition_postcall(sds, node, result)
            if space:
                # XXX maybe try to get something more informative from the
                # processing node (e.g. in 0.5 it used to be 'chunks'->'chunks'
                # to indicate what was trained and what was tested. Now it is
                # more tricky, because `node` could be anything
                result.set_attr(space, (i,))
            # store
            results.append(result)

            if ca.is_enabled("stats") and node.ca.has_key("stats") \
               and node.ca.is_enabled("stats"):
                if not ca.is_set('stats'):
                    # create empty stats container of matching type
                    ca.stats = node.ca['stats'].value.__class__()
                # harvest summary stats
                ca['stats'].value.__iadd__(node.ca['stats'].value)

        # charge condition attribute
        self.ca.repetition_results = results

        # stack all results into a single Dataset
        if concat_as == 'samples':
            results = vstack(results, True)
        elif concat_as == 'features':
            results = hstack(results, True)
        else:
            raise ValueError("Unkown concatenation mode '%s'" % concat_as)
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

        Returns
        -------
        dataset
          The result dataset.
        """
        return result


    def _untrain(self):
        """Untrain this measure and the embedded node."""
        self._node.untrain()
        super(RepeatedMeasure, self)._untrain()


    node = property(fget=lambda self: self._node)
    generator = property(fget=lambda self: self._generator)
    callback = property(fget=lambda self: self._callback)
    concat_as = property(fget=lambda self: self._concat_as)


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
                 splitter=None, **kwargs):
        """
        Parameters
        ----------
        learner : Learner
          Any trainable node that shall be run on the dataset folds.
        generator : Node
          Generator used to resample the input dataset into multiple instances
          (i.e. partitioning it). The number of datasets yielded by this
          generator determines the number of cross-validation folds.
          IMPORTANT: The ``space`` of this generator determines the attribute
          that will be used to split all generated datasets into training and
          testing sets.
        errorfx : Node or callable
          Custom implementation of an error function. The callable needs to
          accept two arguments (1. predicted values, 2. target values).  If not
          a Node, it gets wrapped into a `BinaryFxNode`.
        splitter : Splitter or None
          A Splitter instance to split the dataset into training and testing
          part. The first split will be used for training and the second for
          testing -- all other splits will be ignored. If None, a default
          splitter is auto-generated using the ``space`` setting of the
          ``generator``. The default splitter is configured to return the
          ``1``-labeled partition of the input dataset at first, and the
          ``2``-labeled partition second. This behavior corresponds to most
          Partitioners that label the taken-out portion ``2`` and the remainder
          with ``1``.
        """
        # compile the appropriate repeated measure to do cross-validation from
        # pieces
        if not errorfx is None:
            # error node -- postproc of transfer measure
            if isinstance(errorfx, Node):
                enode = errorfx
            else:
                # wrap into BinaryFxNode
                enode = BinaryFxNode(errorfx, learner.get_space())
        else:
            enode = None

        if splitter is None:
            # default splitter splits into "1" and "2" partition.
            # that will effectively ignore 'deselected' samples (e.g. by
            # Balancer). It is done this way (and not by ignoring '0' samples
            # because it is guaranteed to yield two splits) and is more likely
            # to fail in visible ways if the attribute does not have 0,1,2
            # values at all (i.e. a literal train/test/spareforlater attribute)
            splitter = Splitter(generator.get_space(), attr_values=(1, 2))
        # transfer measure to wrap the learner
        # splitter used the output space of the generator to know what to split
        tm = TransferMeasure(learner, splitter, postproc=enode)

        space = kwargs.pop('space', 'sa.cvfolds')
        # and finally the repeated measure to perform the x-val
        RepeatedMeasure.__init__(self, tm, generator, space=space,
                                 **kwargs)

        for ca in ['stats', 'training_stats']:
            if self.ca.is_enabled(ca):
                # enforce ca if requested
                tm.ca.enable(ca)
        if self.ca.is_enabled('training_stats'):
            # also enable training stats in the learner
            learner.ca.enable('training_stats')

    def __repr__(self, prefixes=[]):
        return super(CrossValidation, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['learner', 'splitter'])
            + _repr_attrs(self, ['errorfx'], default=mean_mismatch_error)
            + _repr_attrs(self, ['space'], default='sa.cvfolds'),
            # Since it is the constructor which generates and passes
            # node=TransferMeasure, it must not be present in __repr__ of CV
            # TODO: clear up hierarchy
            exclude=('node',)
            )


    def _call(self, ds):
        # always untrain to wipe out previous stats
        self.untrain()
        return super(CrossValidation, self)._call(ds)


    def _repetition_postcall(self, ds, node, result):
        # local binding
        ca = self.ca
        if ca.is_enabled("training_stats"):
            if not ca.is_set('training_stats'):
                # create empty stats container of matching type
                ca.training_stats = node.ca['training_stats'].value.__class__()
            # harvest summary stats
            training_stats = node.ca['training_stats'].value
            if isinstance(training_stats, dict):
                # if it was a dictionary of results - we should collect them per item
                for k,v in training_stats.iteritems():
                    if not len(ca['training_stats'].value) or k not in ca['training_stats'].value:
                        ca['training_stats'].value[k] = v
                    else:
                        ca['training_stats'].value[k].__iadd__(v)
            else:
                ca['training_stats'].value.__iadd__(node.ca['training_stats'].value)

        return result


    transfermeasure = property(fget=lambda self:self._node)

    # XXX Well, those properties are defined to match available
    # attributes to constructor arguments.  Unfortunately our
    # hierarchy/API is not ideal at this point
    learner = property(fget=lambda self: self.transfermeasure.measure)
    splitter = property(fget=lambda self: self.transfermeasure.splitter)
    errorfx = property(fget=lambda self: self.transfermeasure.postproc)


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

    is_trained = True
    """Indicate that this measure is always trained."""

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


    def __repr__(self, prefixes=[]):
        return super(TransferMeasure, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['measure', 'splitter'])
            )


    def _call(self, ds):
        # local binding
        measure = self.__measure
        splitter = self.__splitter
        ca = self.ca
        space = self.get_space()

        # generate the training and testing dataset subsequently to reduce the
        # memory footprint, i.e. the splitter might generate copies of the data
        # and no creates one at a time instead of two (for train and test) at
        # once
        # activate the dataset splitter
        dsgen = splitter.generate(ds)
        dstrain = dsgen.next()

        if not len(dstrain):
            raise ValueError(
                "Got empty training dataset from splitting in TransferMeasure. "
                "Unique values of input split attribute are: %s)" \
                % (ds.sa[splitter.get_space()].unique))

        if space:
            # get unique chunks for training set
            train_chunks = ','.join([str(i)
                    for i in dstrain.get_attr(splitter.get_space())[0].unique])
        # ask splitter for first part
        measure.train(dstrain)
        # cleanup to free memory
        del dstrain

        # TODO get training confusion/stats

        # run with second
        dstest = dsgen.next()
        if not len(dstest):
            raise ValueError(
                "Got empty testing dataset from splitting in TransferMeasure. "
                "Unique values of input split attribute are: %s)" \
                % (ds.sa[splitter.get_space()].unique))
        if space:
            # get unique chunks for testing set
            test_chunks = ','.join([str(i)
                    for i in dstest.get_attr(splitter.get_space())[0].unique])
        res = measure(dstest)
        if space:
            # will broadcast to desired length
            res.set_attr(space, ("%s->%s" % (train_chunks, test_chunks),))
        # cleanup to free memory
        del dstest

        # compute measure stats
        if ca.is_enabled('stats'):
            if not hasattr(measure, '__summary_class__'):
                warning('%s has no __summary_class__ attribute -- '
                        'necessary for computing transfer stats' % measure)
            else:
                stats = measure.__summary_class__(
                    # hmm, might be unsupervised, i.e no targets...
                    targets=res.sa[measure.get_space()].value,
                    # XXX this should really accept the full dataset
                    predictions=res.samples[:, 0],
                    estimates=measure.ca.get('estimates', None))
                ca.stats = stats
        if ca.is_enabled('training_stats'):
            if measure.ca.has_key("training_stats") \
               and measure.ca.is_enabled("training_stats"):
                ca.training_stats = measure.ca.training_stats
            else:
                warning("'training_stats' conditional attribute was enabled, "
                        "but the assigned measure '%s' either doesn't support "
                        "it, or it is disabled" % measure)
        return res

    measure = property(fget=lambda self:self.__measure)
    splitter = property(fget=lambda self:self.__splitter)



class FeaturewiseMeasure(Measure):
    """A per-feature-measure computed from a `Dataset` (base class).

    Should behave like a Measure.
    """

    def _postcall(self, dataset, result):
        """Adjusts per-feature-measure for computed `result`
        """
        # This method get the 'result' either as a 1D array, or as a Dataset
        # everything else is illegal.


        if not (len(result.shape) == 1 or isinstance(result, AttrDataset)):
            raise RuntimeError("FeaturewiseMeasures have to return "
                               "their results as 1D array, or as a Dataset "
                               "(error made by: '%s')." % repr(self))

        return Measure._postcall(self, dataset, result)


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

    def __repr__(self, prefixes=[]):
        return super(StaticMeasure, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['measure', 'bias'])
            )

    def _call(self, dataset):
        """Returns assigned sensitivity
        """
        return self.__measure

    #XXX Might need to move into ConditionalAttribute?
    measure = property(fget=lambda self:self.__measure)
    bias = property(fget=lambda self:self.__bias)



def _dont_force_slaves(slave_kwargs={}):
    """Helper to reset force_train in sensitivities with slaves
    """
    # We should not (or even must not in case of SplitCLF) force
    # training of slave analyzers since they would be trained
    # anyways by the Boosted analyzer's train
    # TODO: consider at least a warning whenever it is provided
    # and is True
    slave_kwargs = slave_kwargs or {}   # make new instance of default empty one
    slave_kwargs['force_train'] = slave_kwargs.get('force_train', False)
    return slave_kwargs

#
# Flavored implementations of FeaturewiseMeasures

class Sensitivity(FeaturewiseMeasure):
    """Sensitivities of features for a given Classifier.

    """

    _LEGAL_CLFS = []
    """If Sensitivity is classifier specific, classes of classifiers
    should be listed in the list
    """

    def __init__(self, clf, force_train=True, **kwargs):
        """Initialize the analyzer with the classifier it shall use.

        Parameters
        ----------
        clf : `Classifier`
          classifier to use.
        force_train : bool
          Flag whether the learner will enforce training on the input dataset
          upon every call.
        """

        """Does nothing special."""
        # by default auto train
        kwargs['auto_train'] = kwargs.get('auto_train', True)
        FeaturewiseMeasure.__init__(self, force_train=force_train, **kwargs)

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


    def __repr__(self, prefixes=[]):
        return super(Sensitivity, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['clf'])
            + _repr_attrs(self, ['force_train'], default=True)
            )


    @property
    def is_trained(self):
        return self.__clf.trained

    #    """Train classifier on `dataset` and then compute actual sensitivity.

    #    If the classifier is already trained it is possible to extract the
    #    sensitivities without passing a dataset.
    #    """
    #    # local bindings
    #    clf = self.__clf
    #    if clf.trained:
    #        self._set_trained()
    #    elif self._force_training:
    #        if dataset is None:
    #            raise ValueError, \
    #                  "Training classifier to compute sensitivities requires " \
    #                  "a dataset."
    #        self.train(dataset)

    #    return FeaturewiseMeasure.__call__(self, dataset)


    def _set_classifier(self, clf):
        self.__clf = clf


    def _train(self, dataset):
        clf = self.__clf
        if __debug__:
            debug("SA", "Training classifier %s on %s %s",
                  (clf,
                   dataset,
                   {False: "since it wasn't yet trained",
                    True:  "although it was trained previously"}
                   [clf.trained]))
        return clf.train(dataset)


    def _untrain(self):
        """Untrain corresponding classifier for Sensitivity
        """
        if self.__clf is not None:
            self.__clf.untrain()
        super(Sensitivity, self)._untrain()


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
    def __init__(self, analyzers=None, # XXX should become actually 'measures'
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


    def __repr__(self, prefixes=[]):
        return super(CombinedFeaturewiseMeasure, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['analyzers'])
            + _repr_attrs(self, ['sa_attr'], default='combinations')
            )

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
            smerged = []
            for i, s in enumerate(sensitivities):
                s.sa[sa_attr] = np.repeat(i, len(s))
                smerged.append(s)
            sensitivities = vstack(smerged)
        else:
            sensitivities = \
                Dataset(sensitivities,
                        sa={sa_attr: np.arange(len(sensitivities))})

        self.ca.sensitivities = sensitivities

        return sensitivities


    def _untrain(self):
        """Untrain CombinedFDM
        """
        if self.__analyzers is not None:
            for anal in self.__analyzers:
                anal.untrain()
        super(CombinedFeaturewiseMeasure, self)._untrain()


    ##REF: Name was automagically refactored
    def _set_analyzers(self, analyzers):
        """Set the analyzers
        """
        self.__analyzers = analyzers
        """Analyzers to use"""

    analyzers = property(fget=lambda x:x.__analyzers,
                         fset=_set_analyzers,
                         doc="Used analyzers")



class BoostedClassifierSensitivityAnalyzer(Sensitivity):
    """Set sensitivity analyzers to be merged into a single output"""


    # XXX we might like to pass parameters also for combined_analyzer
    @group_kwargs(prefixes=['slave_'], assign=True)
    def __init__(self,
                 clf,
                 analyzer=None,
                 combined_analyzer=None,
                 sa_attr='lrn_index',
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

        if analyzer is not None and len(self._slave_kwargs):
            raise ValueError, \
                  "Provide either analyzer of slave_* arguments, not both"

        # Do not force_train slave sensitivity since the dataset might
        # be inappropriate -- rely on the classifier being trained by
        # the extraction by the meta classifier itself
        self._slave_kwargs = _dont_force_slaves(self._slave_kwargs)

        if combined_analyzer is None:
            # sanitarize kwargs
            kwargs.pop('force_train', None)
            combined_analyzer = CombinedFeaturewiseMeasure(sa_attr=sa_attr,
                                                                  **kwargs)
        self.__combined_analyzer = combined_analyzer
        """Combined analyzer to use"""

        self.__analyzer = analyzer
        """Analyzer to use for basic classifiers within boosted classifier"""


    ## def __repr__(self, prefixes=[]):
    ##     return super(BoostedClassifierSensitivityAnalyzer, self).__repr__(
    ##         prefixes=prefixes
    ##         + _repr_attrs(self, ['clf', 'analyzer', 'combined_analyzer'])
    ##         + _repr_attrs(self, ['sa_attr'], default='combinations')
    ##         )


    def _untrain(self):
        """Untrain BoostedClassifierSensitivityAnalyzer
        """
        if self.__analyzer is not None:
            self.__analyzer.untrain()
        self.__combined_analyzer.untrain()
        super(BoostedClassifierSensitivityAnalyzer, self)._untrain()


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
                analyzer._force_train = False
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
        # _slave_kwargs is assigned due to assign=True in @group_kwargs
        if analyzer is not None and len(self._slave_kwargs):
            raise ValueError, \
                  "Provide either analyzer of slave_* arguments, not both"

        # Do not force_train slave sensitivity since the dataset might
        # be inappropriate -- rely on the classifier being trained by
        # the extraction by the meta classifier itself
        self._slave_kwargs = _dont_force_slaves(self._slave_kwargs)

        self.__analyzer = analyzer
        """Analyzer to use for basic classifiers within boosted classifier"""


    def _untrain(self):
        super(ProxyClassifierSensitivityAnalyzer, self)._untrain()
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
            analyzer._force_train = False

        result = analyzer._call(dataset)
        self.ca.clf_sensitivities = result

        return result

    analyzer = property(fget=lambda x:x.__analyzer)


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


class FeatureSelectionClassifierSensitivityAnalyzer(ProxyClassifierSensitivityAnalyzer):
    pass

class MappedClassifierSensitivityAnalyzer(ProxyClassifierSensitivityAnalyzer):
    """Set sensitivity analyzer output be reverse mapped using mapper of the
    slave classifier"""

    def _call(self, dataset):
        # incoming dataset need to be forward mapped
        dataset_mapped = self.clf.mapper(dataset)
        if __debug__:
            debug('SA', 'Mapped incoming dataset %s to %s'
                        % (dataset_mapped, dataset))
        sens = super(MappedClassifierSensitivityAnalyzer,
                     self)._call(dataset_mapped)
        return self.clf.mapper.reverse(sens)


    def __str__(self):
        return _str(self, str(self.clf))


