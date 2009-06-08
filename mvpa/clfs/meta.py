# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Classes for meta classifiers -- classifiers which use other classifiers

Meta Classifiers can be grouped according to their function as

:group BoostedClassifiers: CombinedClassifier MulticlassClassifier
  SplitClassifier
:group ProxyClassifiers: ProxyClassifier BinaryClassifier MappedClassifier
  FeatureSelectionClassifier
:group PredictionsCombiners for CombinedClassifier: PredictionsCombiner
  MaximalVote MeanPrediction

"""

__docformat__ = 'restructuredtext'

import operator
import numpy as N

from sets import Set

from mvpa.misc.args import group_kwargs
from mvpa.mappers.mask import MaskMapper
from mvpa.datasets.splitters import NFoldSplitter
from mvpa.misc.state import StateVariable, ClassWithCollections, Harvestable

from mvpa.clfs.base import Classifier
from mvpa.misc.transformers import FirstAxisMean

from mvpa.measures.base import \
    BoostedClassifierSensitivityAnalyzer, ProxyClassifierSensitivityAnalyzer, \
    MappedClassifierSensitivityAnalyzer, \
    FeatureSelectionClassifierSensitivityAnalyzer

from mvpa.base import warning

if __debug__:
    from mvpa.base import debug


class BoostedClassifier(Classifier, Harvestable):
    """Classifier containing the farm of other classifiers.

    Should rarely be used directly. Use one of its childs instead
    """

    # should not be needed if we have prediction_values upstairs
    # raw_predictions should be handled as Harvestable???
    raw_predictions = StateVariable(enabled=False,
        doc="Predictions obtained from each classifier")

    raw_values = StateVariable(enabled=False,
        doc="Values obtained from each classifier")


    def __init__(self, clfs=None, propagate_states=True,
                 harvest_attribs=None, copy_attribs='copy',
                 **kwargs):
        """Initialize the instance.

        :Parameters:
          clfs : list
            list of classifier instances to use (slave classifiers)
          propagate_states : bool
            either to propagate enabled states into slave classifiers.
            It is in effect only when slaves get assigned - so if state
            is enabled not during construction, it would not necessarily
            propagate into slaves
          kwargs : dict
            dict of keyworded arguments which might get used
            by State or Classifier
        """
        if clfs == None:
            clfs = []

        Classifier.__init__(self, **kwargs)
        Harvestable.__init__(self, harvest_attribs, copy_attribs)

        self.__clfs = None
        """Pylint friendly definition of __clfs"""

        self.__propagate_states = propagate_states
        """Enable current enabled states in slave classifiers"""

        self._setClassifiers(clfs)
        """Store the list of classifiers"""


    def __repr__(self, prefixes=[]):
        if self.__clfs is None or len(self.__clfs)==0:
            #prefix_ = "clfs=%s" % repr(self.__clfs)
            prefix_ = []
        else:
            prefix_ = ["clfs=[%s,...]" % repr(self.__clfs[0])]
        return super(BoostedClassifier, self).__repr__(prefix_ + prefixes)


    def _train(self, dataset):
        """Train `BoostedClassifier`
        """
        for clf in self.__clfs:
            clf.train(dataset)


    def _posttrain(self, dataset):
        """Custom posttrain of `BoostedClassifier`

        Harvest over the trained classifiers if it was asked to so
        """
        Classifier._posttrain(self, dataset)
        if self.states.isEnabled('harvested'):
            for clf in self.__clfs:
                self._harvest(locals())
        if self.params.retrainable:
            self.__changedData_isset = False


    def _getFeatureIds(self):
        """Custom _getFeatureIds for `BoostedClassifier`
        """
        # return union of all used features by slave classifiers
        feature_ids = Set([])
        for clf in self.__clfs:
            feature_ids = feature_ids.union(Set(clf.feature_ids))
        return list(feature_ids)


    def _predict(self, data):
        """Predict using `BoostedClassifier`
        """
        raw_predictions = [ clf.predict(data) for clf in self.__clfs ]
        self.raw_predictions = raw_predictions
        assert(len(self.__clfs)>0)
        if self.states.isEnabled("values"):
            if N.array([x.states.isEnabled("values")
                        for x in self.__clfs]).all():
                values = [ clf.values for clf in self.__clfs ]
                self.raw_values = values
            else:
                warning("One or more classifiers in %s has no 'values' state" %
                        self + "enabled, thus BoostedClassifier can't have" +
                        " 'raw_values' state variable defined")

        return raw_predictions


    def _setClassifiers(self, clfs):
        """Set the classifiers used by the boosted classifier

        We have to allow to set list of classifiers after the object
        was actually created. It will be used by
        MulticlassClassifier
        """
        self.__clfs = clfs
        """Classifiers to use"""

        if len(clfs):
            for flag in ['regression']:
                values = N.array([clf.params[flag].value for clf in clfs])
                value = values.any()
                if __debug__:
                    debug("CLFBST", "Setting %(flag)s=%(value)s for classifiers "
                          "%(clfs)s with %(values)s",
                          msgargs={'flag' : flag, 'value' : value,
                                   'clfs' : clfs,
                                   'values' : values})
                # set flag if it needs to be trained before predicting
                self.params[flag].value = value

            # enable corresponding states in the slave-classifiers
            if self.__propagate_states:
                for clf in self.__clfs:
                    clf.states.enable(self.states.enabled, missingok=True)

        # adhere to their capabilities + 'multiclass'
        # XXX do intersection across all classifiers!
        # TODO: this seems to be wrong since it can be regression etc
        self._clf_internals = [ 'binary', 'multiclass', 'meta' ]
        if len(clfs)>0:
            self._clf_internals += self.__clfs[0]._clf_internals

    def untrain(self):
        """Untrain `BoostedClassifier`

        Has to untrain any known classifier
        """
        if not self.trained:
            return
        for clf in self.clfs:
            clf.untrain()
        super(BoostedClassifier, self).untrain()

    def getSensitivityAnalyzer(self, **kwargs):
        """Return an appropriate SensitivityAnalyzer"""
        return BoostedClassifierSensitivityAnalyzer(
                self,
                **kwargs)


    clfs = property(fget=lambda x:x.__clfs,
                    fset=_setClassifiers,
                    doc="Used classifiers")



class ProxyClassifier(Classifier):
    """Classifier which decorates another classifier

    Possible uses:

     - modify data somehow prior training/testing:
       * normalization
       * feature selection
       * modification

     - optimized classifier?

    """

    def __init__(self, clf, **kwargs):
        """Initialize the instance

        :Parameters:
          clf : Classifier
            classifier based on which mask classifiers is created
          """

        Classifier.__init__(self, regression=clf.regression, **kwargs)

        self.__clf = clf
        """Store the classifier to use."""

        # adhere to slave classifier capabilities
        # TODO: unittest
        self._clf_internals = self._clf_internals[:] + ['meta']
        if clf is not None:
            self._clf_internals += clf._clf_internals


    def __repr__(self, prefixes=[]):
        return super(ProxyClassifier, self).__repr__(
            ["clf=%s" % repr(self.__clf)] + prefixes)

    def summary(self):
        s = super(ProxyClassifier, self).summary()
        if self.trained:
            s += "\n Slave classifier summary:" + \
                 '\n + %s' % \
                 (self.__clf.summary().replace('\n', '\n |'))
        return s



    def _train(self, dataset):
        """Train `ProxyClassifier`
        """
        # base class does nothing much -- just proxies requests to underlying
        # classifier
        self.__clf.train(dataset)

        # for the ease of access
        # TODO: if to copy we should exclude some states which are defined in
        #       base Classifier (such as training_time, predicting_time)
        # YOH: for now _copy_states_ would copy only set states variables. If
        #      anything needs to be overriden in the parent's class, it is
        #      welcome to do so
        #self.states._copy_states_(self.__clf, deep=False)


    def _predict(self, data):
        """Predict using `ProxyClassifier`
        """
        clf = self.__clf
        if self.states.isEnabled('values'):
            clf.states.enable(['values'])

        result = clf.predict(data)
        # for the ease of access
        self.states._copy_states_(self.__clf, ['values'], deep=False)
        return result


    def untrain(self):
        """Untrain ProxyClassifier
        """
        if not self.__clf is None:
            self.__clf.untrain()
        super(ProxyClassifier, self).untrain()


    @group_kwargs(prefixes=['slave_'], passthrough=True)
    def getSensitivityAnalyzer(self, slave_kwargs, **kwargs):
        """Return an appropriate SensitivityAnalyzer"""
        return ProxyClassifierSensitivityAnalyzer(
                self,
                analyzer=self.__clf.getSensitivityAnalyzer(**slave_kwargs),
                **kwargs)


    clf = property(lambda x:x.__clf, doc="Used `Classifier`")



#
# Various combiners for CombinedClassifier
#

class PredictionsCombiner(ClassWithCollections):
    """Base class for combining decisions of multiple classifiers"""

    def train(self, clfs, dataset):
        """PredictionsCombiner might need to be trained

        :Parameters:
          clfs : list of Classifier
            List of classifiers to combine. Has to be classifiers (not
            pure predictions), since combiner might use some other
            state variables (value's) instead of pure prediction's
          dataset : Dataset
            training data in this case
        """
        pass


    def __call__(self, clfs, dataset):
        """Call function

        :Parameters:
          clfs : list of Classifier
            List of classifiers to combine. Has to be classifiers (not
            pure predictions), since combiner might use some other
            state variables (value's) instead of pure prediction's
        """
        raise NotImplementedError



class MaximalVote(PredictionsCombiner):
    """Provides a decision using maximal vote rule"""

    predictions = StateVariable(enabled=True,
        doc="Voted predictions")
    all_label_counts = StateVariable(enabled=False,
        doc="Counts across classifiers for each label/sample")

    def __init__(self):
        """XXX Might get a parameter to use raw decision values if
        voting is not unambigous (ie two classes have equal number of
        votes
        """
        PredictionsCombiner.__init__(self)


    def __call__(self, clfs, dataset):
        """Actuall callable - perform voting

        Extended functionality which might not be needed actually:
        Since `BinaryClassifier` might return a list of possible
        predictions (not just a single one), we should consider all of those

        MaximalVote doesn't care about dataset itself
        """
        if len(clfs)==0:
            return []                   # to don't even bother

        all_label_counts = None
        for clf in clfs:
            # Lets check first if necessary state variable is enabled
            if not clf.states.isEnabled("predictions"):
                raise ValueError, "MaximalVote needs classifiers (such as " + \
                      "%s) with state 'predictions' enabled" % clf
            predictions = clf.predictions
            if all_label_counts is None:
                all_label_counts = [ {} for i in xrange(len(predictions)) ]

            # for every sample
            for i in xrange(len(predictions)):
                prediction = predictions[i]
                if not operator.isSequenceType(prediction):
                    prediction = (prediction,)
                for label in prediction: # for every label
                    # XXX we might have multiple labels assigned
                    # but might not -- don't remember now
                    if not all_label_counts[i].has_key(label):
                        all_label_counts[i][label] = 0
                    all_label_counts[i][label] += 1

        predictions = []
        # select maximal vote now for each sample
        for i in xrange(len(all_label_counts)):
            label_counts = all_label_counts[i]
            # lets do explicit search for max so we know
            # if it is unique
            maxk = []                   # labels of elements with max vote
            maxv = -1
            for k, v in label_counts.iteritems():
                if v > maxv:
                    maxk = [k]
                    maxv = v
                elif v == maxv:
                    maxk.append(k)

            assert len(maxk) >= 1, \
                   "We should have obtained at least a single key of max label"

            if len(maxk) > 1:
                warning("We got multiple labels %s which have the " % maxk +
                        "same maximal vote %d. XXX disambiguate" % maxv)
            predictions.append(maxk[0])

        self.all_label_counts = all_label_counts
        self.predictions = predictions
        return predictions



class MeanPrediction(PredictionsCombiner):
    """Provides a decision by taking mean of the results
    """

    predictions = StateVariable(enabled=True,
        doc="Mean predictions")

    def __call__(self, clfs, dataset):
        """Actuall callable - perform meaning

        """
        if len(clfs)==0:
            return []                   # to don't even bother

        all_predictions = []
        for clf in clfs:
            # Lets check first if necessary state variable is enabled
            if not clf.states.isEnabled("predictions"):
                raise ValueError, "MeanPrediction needs classifiers (such " \
                      " as %s) with state 'predictions' enabled" % clf
            all_predictions.append(clf.predictions)

        # compute mean
        predictions = N.mean(N.asarray(all_predictions), axis=0)
        self.predictions = predictions
        return predictions


class ClassifierCombiner(PredictionsCombiner):
    """Provides a decision using training a classifier on predictions/values

    TODO: implement
    """

    predictions = StateVariable(enabled=True,
        doc="Trained predictions")


    def __init__(self, clf, variables=None):
        """Initialize `ClassifierCombiner`

        :Parameters:
          clf : Classifier
            Classifier to train on the predictions
          variables : list of basestring
            List of state variables stored in 'combined' classifiers, which
            to use as features for training this classifier
        """
        PredictionsCombiner.__init__(self)

        self.__clf = clf
        """Classifier to train on `variables` states of provided classifiers"""

        if variables == None:
            variables = ['predictions']
        self.__variables = variables
        """What state variables of the classifiers to use"""


    def untrain(self):
        """It might be needed to untrain used classifier"""
        if self.__clf:
            self.__clf.untrain()

    def __call__(self, clfs, dataset):
        """
        """
        if len(clfs)==0:
            return []                   # to don't even bother

        raise NotImplementedError



class CombinedClassifier(BoostedClassifier):
    """`BoostedClassifier` which combines predictions using some
    `PredictionsCombiner` functor.
    """

    def __init__(self, clfs=None, combiner=None, **kwargs):
        """Initialize the instance.

        :Parameters:
          clfs : list of Classifier
            list of classifier instances to use
          combiner : PredictionsCombiner
            callable which takes care about combining multiple
            results into a single one (e.g. maximal vote for
            classification, MeanPrediction for regression))
          kwargs : dict
            dict of keyworded arguments which might get used
            by State or Classifier

        NB: `combiner` might need to operate not on 'predictions' descrete
            labels but rather on raw 'class' values classifiers
            estimate (which is pretty much what is stored under
            `values`
        """
        if clfs == None:
            clfs = []

        BoostedClassifier.__init__(self, clfs, **kwargs)

        # assign default combiner
        if combiner is None:
            combiner = (MaximalVote, MeanPrediction)[int(self.regression)]()
        self.__combiner = combiner
        """Functor destined to combine results of multiple classifiers"""


    def __repr__(self, prefixes=[]):
        """Literal representation of `CombinedClassifier`.
        """
        return super(CombinedClassifier, self).__repr__(
            ["combiner=%s" % repr(self.__combiner)] + prefixes)


    def summary(self):
        """Provide summary for the `CombinedClassifier`.
        """
        s = super(CombinedClassifier, self).summary()
        if self.trained:
            s += "\n Slave classifiers summaries:"
            for i, clf in enumerate(self.clfs):
                s += '\n + %d clf: %s' % \
                     (i, clf.summary().replace('\n', '\n |'))
        return s


    def untrain(self):
        """Untrain `CombinedClassifier`
        """
        try:
            self.__combiner.untrain()
        except:
            pass
        super(CombinedClassifier, self).untrain()

    def _train(self, dataset):
        """Train `CombinedClassifier`
        """
        BoostedClassifier._train(self, dataset)
        # combiner might need to train as well
        self.__combiner.train(self.clfs, dataset)


    def _predict(self, data):
        """Predict using `CombinedClassifier`
        """
        BoostedClassifier._predict(self, data)
        # combiner will make use of state variables instead of only predictions
        # returned from _predict
        predictions = self.__combiner(self.clfs, data)
        self.predictions = predictions

        if self.states.isEnabled("values"):
            if self.__combiner.states.isActive("values"):
                # XXX or may be we could leave simply up to accessing .combiner?
                self.values = self.__combiner.values
            else:
                if __debug__:
                    warning("Boosted classifier %s has 'values' state enabled,"
                            " but combiner doesn't have 'values' active, thus "
                            " .values cannot be provided directly, access .clfs"
                            % self)
        return predictions


    combiner = property(fget=lambda x:x.__combiner,
                        doc="Used combiner to derive a single result")



class BinaryClassifier(ProxyClassifier):
    """`ProxyClassifier` which maps set of two labels into +1 and -1
    """

    def __init__(self, clf, poslabels, neglabels, **kwargs):
        """
        :Parameters:
          clf : Classifier
            classifier to use
          poslabels : list
            list of labels which are treated as +1 category
          neglabels : list
            list of labels which are treated as -1 category
        """

        ProxyClassifier.__init__(self, clf, **kwargs)

        self._regressionIsBogus()

        # Handle labels
        sposlabels = Set(poslabels) # so to remove duplicates
        sneglabels = Set(neglabels) # so to remove duplicates

        # check if there is no overlap
        overlap = sposlabels.intersection(sneglabels)
        if len(overlap)>0:
            raise ValueError("Sets of positive and negative labels for " +
                "BinaryClassifier must not overlap. Got overlap " %
                overlap)

        self.__poslabels = list(sposlabels)
        self.__neglabels = list(sneglabels)

        # define what values will be returned by predict: if there is
        # a single label - return just it alone, otherwise - whole
        # list
        # Such approach might come useful if we use some classifiers
        # over different subsets of data with some voting later on
        # (1-vs-therest?)

        if len(self.__poslabels) > 1:
            self.__predictpos = self.__poslabels
        else:
            self.__predictpos = self.__poslabels[0]

        if len(self.__neglabels) > 1:
            self.__predictneg = self.__neglabels
        else:
            self.__predictneg = self.__neglabels[0]


    def __repr__(self, prefixes=[]):
        prefix = "poslabels=%s, neglabels=%s" % (
            repr(self.__poslabels), repr(self.__neglabels))
        return super(BinaryClassifier, self).__repr__([prefix] + prefixes)


    def _train(self, dataset):
        """Train `BinaryClassifier`
        """
        idlabels = [(x, +1) for x in dataset.idsbylabels(self.__poslabels)] + \
                    [(x, -1) for x in dataset.idsbylabels(self.__neglabels)]
        # XXX we have to sort ids since at the moment Dataset.selectSamples
        #     doesn't take care about order
        idlabels.sort()
        # select the samples
        orig_labels = None

        # If we need all samples, why simply not perform on original
        # data, an just store/restore labels. But it really should be done
        # within Dataset.selectSamples
        if len(idlabels) == dataset.nsamples \
            and [x[0] for x in idlabels] == range(dataset.nsamples):
            # the last condition is not even necessary... just overly
            # cautious
            datasetselected = dataset   # no selection is needed
            orig_labels = dataset.labels # but we would need to restore labels
            if __debug__:
                debug('CLFBIN',
                      "Assigned all %d samples for binary " %
                      (dataset.nsamples) +
                      " classification among labels %s/+1 and %s/-1" %
                      (self.__poslabels, self.__neglabels))
        else:
            datasetselected = dataset.selectSamples([ x[0] for x in idlabels ])
            if __debug__:
                debug('CLFBIN',
                      "Selected %d samples out of %d samples for binary " %
                      (len(idlabels), dataset.nsamples) +
                      " classification among labels %s/+1 and %s/-1" %
                      (self.__poslabels, self.__neglabels) +
                      ". Selected %s" % datasetselected)

        # adjust the labels
        datasetselected.labels = [ x[1] for x in idlabels ]

        # now we got a dataset with only 2 labels
        if __debug__:
            assert((datasetselected.uniquelabels == [-1, 1]).all())

        self.clf.train(datasetselected)

        if not orig_labels is None:
            dataset.labels = orig_labels

    def _predict(self, data):
        """Predict the labels for a given `data`

        Predicts using binary classifier and spits out list (for each sample)
        where with either poslabels or neglabels as the "label" for the sample.
        If there was just a single label within pos or neg labels then it would
        return not a list but just that single label.
        """
        binary_predictions = ProxyClassifier._predict(self, data)
        self.values = binary_predictions
        predictions = [ {-1: self.__predictneg,
                         +1: self.__predictpos}[x] for x in binary_predictions]
        self.predictions = predictions
        return predictions



class MulticlassClassifier(CombinedClassifier):
    """`CombinedClassifier` to perform multiclass using a list of
    `BinaryClassifier`.

    such as 1-vs-1 (ie in pairs like libsvm doesn) or 1-vs-all (which
    is yet to think about)
    """

    def __init__(self, clf, bclf_type="1-vs-1", **kwargs):
        """Initialize the instance

        :Parameters:
          clf : Classifier
            classifier based on which multiple classifiers are created
            for multiclass
          bclf_type
            "1-vs-1" or "1-vs-all", determines the way to generate binary
            classifiers
          """
        CombinedClassifier.__init__(self, **kwargs)
        self._regressionIsBogus()
        if not clf is None:
            clf._regressionIsBogus()

        self.__clf = clf
        """Store sample instance of basic classifier"""

        # Some checks on known ways to do multiclass
        if bclf_type == "1-vs-1":
            pass
        elif bclf_type == "1-vs-all": # TODO
            raise NotImplementedError
        else:
            raise ValueError, \
                  "Unknown type of classifier %s for " % bclf_type + \
                  "BoostedMulticlassClassifier"
        self.__bclf_type = bclf_type

    # XXX fix it up a bit... it seems that MulticlassClassifier should
    # be actually ProxyClassifier and use BoostedClassifier internally
    def __repr__(self, prefixes=[]):
        prefix = "bclf_type=%s, clf=%s" % (repr(self.__bclf_type),
                                            repr(self.__clf))
        return super(MulticlassClassifier, self).__repr__([prefix] + prefixes)


    def _train(self, dataset):
        """Train classifier
        """
        # construct binary classifiers
        ulabels = dataset.uniquelabels
        if self.__bclf_type == "1-vs-1":
            # generate pairs and corresponding classifiers
            biclfs = []
            for i in xrange(len(ulabels)):
                for j in xrange(i+1, len(ulabels)):
                    clf = self.__clf.clone()
                    biclfs.append(
                        BinaryClassifier(
                            clf,
                            poslabels=[ulabels[i]], neglabels=[ulabels[j]]))
            if __debug__:
                debug("CLFMC", "Created %d binary classifiers for %d labels" %
                      (len(biclfs), len(ulabels)))

            self.clfs = biclfs

        elif self.__bclf_type == "1-vs-all":
            raise NotImplementedError

        # perform actual training
        CombinedClassifier._train(self, dataset)



class SplitClassifier(CombinedClassifier):
    """`BoostedClassifier` to work on splits of the data

    """

    """
    TODO: SplitClassifier and MulticlassClassifier have too much in
          common -- need to refactor: just need a splitter which would
          split dataset in pairs of class labels. MulticlassClassifier
          does just a tiny bit more which might be not necessary at
          all: map sets of labels into 2 categories...
    """

    # TODO: unify with CrossValidatedTransferError which now uses
    # harvest_attribs to expose gathered attributes
    confusion = StateVariable(enabled=False,
        doc="Resultant confusion whenever classifier trained " +
            "on 1 part and tested on 2nd part of each split")

    splits = StateVariable(enabled=False, doc=
       """Store the actual splits of the data. Can be memory expensive""")

    # ??? couldn't be training_confusion since it has other meaning
    #     here, BUT it is named so within CrossValidatedTransferError
    #     -- unify
    #  decided to go with overriding semantics tiny bit. For split
    #     classifier training_confusion would correspond to summary
    #     over training errors across all splits. Later on if need comes
    #     we might want to implement global_training_confusion which would
    #     correspond to overall confusion on full training dataset as it is
    #     done in base Classifier
    #global_training_confusion = StateVariable(enabled=False,
    #    doc="Summary over training confusions acquired at each split")

    def __init__(self, clf, splitter=NFoldSplitter(cvtype=1), **kwargs):
        """Initialize the instance

        :Parameters:
          clf : Classifier
            classifier based on which multiple classifiers are created
            for multiclass
          splitter : Splitter
            `Splitter` to use to split the dataset prior training
          """

        CombinedClassifier.__init__(self, regression=clf.regression, **kwargs)
        self.__clf = clf
        """Store sample instance of basic classifier"""

        if isinstance(splitter, type):
            raise ValueError, \
                  "Please provide an instance of a splitter, not a type." \
                  " Got %s" % splitter

        self.__splitter = splitter


    def _train(self, dataset):
        """Train `SplitClassifier`
        """
        # generate pairs and corresponding classifiers
        bclfs = []

        # local binding
        states = self.states

        clf_template = self.__clf
        if states.isEnabled('confusion'):
            states.confusion = clf_template._summaryClass()
        if states.isEnabled('training_confusion'):
            clf_template.states.enable(['training_confusion'])
            states.training_confusion = clf_template._summaryClass()

        clf_hastestdataset = hasattr(clf_template, 'testdataset')

        # for proper and easier debugging - first define classifiers and then
        # train them
        for split in self.__splitter.splitcfg(dataset):
            if __debug__:
                debug("CLFSPL_",
                      "Deepcopying %(clf)s for %(sclf)s",
                      msgargs={'clf':clf_template,
                               'sclf':self})
            clf = clf_template.clone()
            bclfs.append(clf)
        self.clfs = bclfs

        self.splits = []

        for i, split in enumerate(self.__splitter(dataset)):
            if __debug__:
                debug("CLFSPL", "Training classifier for split %d" % (i))

            if states.isEnabled("splits"):
                self.splits.append(split)

            clf = self.clfs[i]

            # assign testing dataset if given classifier can digest it
            if clf_hastestdataset:
                clf.testdataset = split[1]

            clf.train(split[0])

            # unbind the testdataset from the classifier
            if clf_hastestdataset:
                clf.testdataset = None

            if states.isEnabled("confusion"):
                predictions = clf.predict(split[1].samples)
                self.confusion.add(split[1].labels, predictions,
                                   clf.states.get('values', None))
                if __debug__:
                    dact = debug.active
                    if 'CLFSPL_' in dact:
                        debug('CLFSPL_', 'Split %d:\n%s' % (i, self.confusion))
                    elif 'CLFSPL' in dact:
                        debug('CLFSPL', 'Split %d error %.2f%%'
                              % (i, self.confusion.summaries[-1].error))

            if states.isEnabled("training_confusion"):
                states.training_confusion += \
                                               clf.states.training_confusion
        # hackish way -- so it should work only for ConfusionMatrix???
        try:
            if states.isEnabled("confusion"):
                states.confusion.labels_map = dataset.labels_map
            if states.isEnabled("training_confusion"):
                states.training_confusion.labels_map = dataset.labels_map
        except:
            pass


    @group_kwargs(prefixes=['slave_'], passthrough=True)
    def getSensitivityAnalyzer(self, slave_kwargs, **kwargs):
        """Return an appropriate SensitivityAnalyzer for `SplitClassifier`

        :Parameters:
          combiner
            If not provided, FirstAxisMean is assumed
        """
        kwargs.setdefault('combiner', FirstAxisMean)
        return BoostedClassifierSensitivityAnalyzer(
                self,
                analyzer=self.__clf.getSensitivityAnalyzer(**slave_kwargs),
                **kwargs)

    splitter = property(fget=lambda x:x.__splitter,
                        doc="Splitter user by SplitClassifier")


class MappedClassifier(ProxyClassifier):
    """`ProxyClassifier` which uses some mapper prior training/testing.

    `MaskMapper` can be used just a subset of features to
    train/classify.
    Having such classifier we can easily create a set of classifiers
    for BoostedClassifier, where each classifier operates on some set
    of features, e.g. set of best spheres from SearchLight, set of
    ROIs selected elsewhere. It would be different from simply
    applying whole mask over the dataset, since here initial decision
    is made by each classifier and then later on they vote for the
    final decision across the set of classifiers.
    """

    def __init__(self, clf, mapper, **kwargs):
        """Initialize the instance

        :Parameters:
          clf : Classifier
            classifier based on which mask classifiers is created
          mapper
            whatever `Mapper` comes handy
          """
        ProxyClassifier.__init__(self, clf, **kwargs)

        self.__mapper = mapper
        """mapper to help us our with prepping data to
        training/classification"""


    def _train(self, dataset):
        """Train `MappedClassifier`
        """
        # first train the mapper
        # XXX: should training be done using whole dataset or just samples
        # YYY: in some cases labels might be needed, thus better full dataset
        self.__mapper.train(dataset)

        # for train() we have to provide dataset -- not just samples to train!
        wdataset = dataset.applyMapper(featuresmapper = self.__mapper)
        ProxyClassifier._train(self, wdataset)


    def _predict(self, data):
        """Predict using `MappedClassifier`
        """
        return ProxyClassifier._predict(self, self.__mapper.forward(data))


    @group_kwargs(prefixes=['slave_'], passthrough=True)
    def getSensitivityAnalyzer(self, slave_kwargs, **kwargs):
        """Return an appropriate SensitivityAnalyzer"""
        return MappedClassifierSensitivityAnalyzer(
                self,
                analyzer=self.clf.getSensitivityAnalyzer(**slave_kwargs),
                **kwargs)


    mapper = property(lambda x:x.__mapper, doc="Used mapper")



class FeatureSelectionClassifier(ProxyClassifier):
    """`ProxyClassifier` which uses some `FeatureSelection` prior training.

    `FeatureSelection` is used first to select features for the classifier to
    use for prediction. Internally it would rely on MappedClassifier which
    would use created MaskMapper.

    TODO: think about removing overhead of retraining the same classifier if
    feature selection was carried out with the same classifier already. It
    has been addressed by adding .trained property to classifier, but now
    we should expclitely use isTrained here if we want... need to think more
    """

    _clf_internals = [ 'does_feature_selection', 'meta' ]

    def __init__(self, clf, feature_selection, testdataset=None, **kwargs):
        """Initialize the instance

        :Parameters:
          clf : Classifier
            classifier based on which mask classifiers is created
          feature_selection : FeatureSelection
            whatever `FeatureSelection` comes handy
          testdataset : Dataset
            optional dataset which would be given on call to feature_selection
          """
        ProxyClassifier.__init__(self, clf, **kwargs)

        self.__maskclf = None
        """Should become `MappedClassifier`(mapper=`MaskMapper`) later on."""

        self.__feature_selection = feature_selection
        """`FeatureSelection` to select the features prior training"""

        self.__testdataset = testdataset
        """`FeatureSelection` might like to use testdataset"""


    def untrain(self):
        """Untrain `FeatureSelectionClassifier`

        Has to untrain any known classifier
        """
        if not self.trained:
            return
        if not self.__maskclf is None:
            self.__maskclf.untrain()
        super(FeatureSelectionClassifier, self).untrain()


    def _train(self, dataset):
        """Train `FeatureSelectionClassifier`
        """
        # temporarily enable selected_ids
        self.__feature_selection.states._changeTemporarily(
            enable_states=["selected_ids"])

        if __debug__:
            debug("CLFFS", "Performing feature selection using %s" %
                  self.__feature_selection + " on %s" % dataset)

        (wdataset, tdataset) = self.__feature_selection(dataset,
                                                        self.__testdataset)
        if __debug__:
            add_ = ""
            if "CLFFS_" in debug.active:
                add_ = " Selected features: %s" % \
                       self.__feature_selection.selected_ids
            debug("CLFFS", "%(fs)s selected %(nfeat)d out of " +
                  "%(dsnfeat)d features.%(app)s",
                  msgargs={'fs':self.__feature_selection,
                           'nfeat':wdataset.nfeatures,
                           'dsnfeat':dataset.nfeatures,
                           'app':add_})

        # create a mask to devise a mapper
        # TODO -- think about making selected_ids a MaskMapper
        mappermask = N.zeros(dataset.nfeatures)
        mappermask[self.__feature_selection.selected_ids] = 1
        mapper = MaskMapper(mappermask)

        self.__feature_selection.states._resetEnabledTemporarily()

        # create and assign `MappedClassifier`
        self.__maskclf = MappedClassifier(self.clf, mapper)
        # we could have called self.__clf.train(dataset), but it would
        # cause unnecessary masking
        self.__maskclf.clf.train(wdataset)

        # for the ease of access
        # TODO see for ProxyClassifier
        #self.states._copy_states_(self.__maskclf, deep=False)

    def _getFeatureIds(self):
        """Return used feature ids for `FeatureSelectionClassifier`

        """
        return self.__feature_selection.selected_ids

    def _predict(self, data):
        """Predict using `FeatureSelectionClassifier`
        """
        clf = self.__maskclf
        if self.states.isEnabled('values'):
            clf.states.enable(['values'])

        result = clf._predict(data)
        # for the ease of access
        self.states._copy_states_(clf, ['values'], deep=False)
        return result

    def setTestDataset(self, testdataset):
        """Set testing dataset to be used for feature selection
        """
        self.__testdataset = testdataset

    maskclf = property(lambda x:x.__maskclf, doc="Used `MappedClassifier`")
    feature_selection = property(lambda x:x.__feature_selection,
                                 doc="Used `FeatureSelection`")

    @group_kwargs(prefixes=['slave_'], passthrough=True)
    def getSensitivityAnalyzer(self, slave_kwargs, **kwargs):
        """Return an appropriate SensitivityAnalyzer

        had to clone from mapped classifier???
        """
        return FeatureSelectionClassifierSensitivityAnalyzer(
                self,
                analyzer=self.clf.getSensitivityAnalyzer(**slave_kwargs),
                **kwargs)



    testdataset = property(fget=lambda x:x.__testdataset,
                           fset=setTestDataset)
