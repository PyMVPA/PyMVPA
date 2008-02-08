#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Base classes for all classifiers.

Base Classifiers can be grouped according to their function as

:group Basic Classifiers: Classifier BoostedClassifier ProxyClassifier
:group BoostedClassifiers: CombinedClassifier MulticlassClassifier
  SplitClassifier
:group ProxyClassifiers: BinaryClassifier MappedClassifier
  FeatureSelectionClassifier
:group PredictionsCombiners for CombinedClassifier: PredictionsCombiner MaximalVote

"""

__docformat__ = 'restructuredtext'

import operator
import numpy as N

from copy import deepcopy
from sets import Set

from mvpa.datasets.maskmapper import MaskMapper
from mvpa.datasets.splitter import NFoldSplitter
from mvpa.misc.state import StateVariable, Stateful

from transerror import ConfusionMatrix

from mvpa.misc import warning

if __debug__:
    import traceback
    from mvpa.misc import debug


def _deepcopyclf(clf):
    """Deepcopying of a classifier.

    If deepcopy fails -- tries to untrain it first so that there is no
    swig bindings attached
    """
    try:
        return deepcopy(clf)
    except:
        clf.untrain()
        return deepcopy(clf)


class Classifier(Stateful):
    """Abstract classifier class to be inherited by all classifiers

    Required behavior:

    For every classifier is has to be possible to be instanciated without
    having to specify the training pattern.

    Repeated calls to the train() method with different training data have to
    result in a valid classifier, trained for the particular dataset.

    It must be possible to specify all classifier parameters as keyword
    arguments to the constructor.

    Recommended behavior:

    Derived classifiers should provide access to *values* -- i.e. that
    information that is finally used to determine the predicted class label.

    Michael: Maybe it works well if each classifier provides a 'values'
             state member. This variable is a list as long as and in same order
             as Dataset.uniquelabels (training data). Each item in the list
             corresponds to the likelyhood of a sample to belong to the
             respective class. However the sematics might differ between
             classifiers, e.g. kNN would probably store distances to class-
             neighbours, where PLF would store the raw function value of the
             logistic function. So in the case of kNN low is predictive and for
             PLF high is predictive. Don't know if there is the need to unify
             that.

             As the storage and/or computation of this information might be
             demanding its collection should be switchable and off be default.

    Nomenclature
     * predictions  : corresponds to the quantized labels if classifier spits
                      out labels by .predict()
     * values : might be different from predictions if a classifier's predict()
                   makes a decision based on some internal value such as
                   probability or a distance.
    """
    # Dict that contains the parameters of a classifier.
    # This shall provide an interface to plug generic parameter optimizer
    # on all classifiers (e.g. grid- or line-search optimizer)
    # A dictionary is used because Michael thinks that access by name is nicer.
    # Additonally Michael thinks ATM that additonal information might be
    # necessary in some situations (e.g. reasonably predefined parameter range,
    # minimal iteration stepsize, ...), therefore the value to each key should
    # also be a dict or we should use mvpa.misc.param.Parameter'...

    trained_labels = StateVariable(enabled=True,
        doc="What labels (unique) clf was trained on")

    training_confusion = StateVariable(enabled=True,
        doc="Result of learning: `ConfusionMatrix` " \
            "(and corresponding learning error)")

    predictions = StateVariable(enabled=True,
        doc="Reported predicted values")

    values = StateVariable(enabled=False,
        doc="Internal values seen by the classifier")


    params = {}

    def __init__(self, train2predict=True, **kwargs):
        """Cheap initialization.
        """
        Stateful.__init__(self, **kwargs)

        self.__train2predict = train2predict
        """Some classifiers might not need to be trained to predict"""

        self.__trainednfeatures = None
        """Stores number of features for which classifier was trained.
        If None -- it wasn't trained at all"""


        self.__trainedid = None
        """Stores id of the dataset on which it was trained to signal
        in trained() if it was trained already on the same dataset"""


    def __str__(self):
        return "%s\n %s" % (`self`, Stateful.__str__(self))


    def _pretrain(self, dataset):
        """Functionality prior to training
        """
        pass


    def _posttrain(self, dataset, result):
        """Functionality post training

        For instance -- computing confusion matrix
        """
        self.trained_labels = Set(dataset.uniquelabels)

        # needs to be assigned first since below we use predict
        self.__trainednfeatures = dataset.nfeatures
        self.__trainedid = dataset._id
        if self.states.isEnabled('training_confusion'):
            # we should not store predictions for training data,
            # it is confusing imho (yoh)
            self.states._changeTemporarily(
                disable_states=["predictions"])
            predictions = self.predict(dataset.samples)
            self.states._resetEnabledTemporarily()
            self.training_confusion = ConfusionMatrix(
                labels=dataset.uniquelabels, targets=dataset.labels,
                predictions=predictions)


    def _train(self, dataset):
        """Function to be actually overriden in derived classes
        """
        raise NotImplementedError


    def train(self, dataset):
        """Train classifier on a dataset

        Shouldn't be overriden in subclasses unless explicitely needed
        to do so
        """
        if __debug__:
            debug("CLF", "Training classifier %s on dataset %s" % \
                  (`self`, `dataset`))
            tb = traceback.extract_stack(limit=5)
            debug("CLF_TB", "Traceback: %s" % tb)

        self._pretrain(dataset)
        result = self._train(dataset)
        self._posttrain(dataset, result)
        return result


    def _prepredict(self, data):
        """Functionality prior prediction
        """
        if self.__train2predict:
            # check if classifier was trained if that is needed
            if not self.trained:
                raise ValueError, \
                      "Classifier %s wasn't yet trained, therefore can't " \
                      "predict" % `self`
            nfeatures = data.shape[1]
            # check if number of features is the same as in the data
            # it was trained on
            if nfeatures != self.__trainednfeatures:
                raise ValueError, \
                      "Classifier %s was trained on data with %d features, " % \
                      (`self`, self.__trainednfeatures) + \
                      "thus can't predict for %d features" % nfeatures


    def _postpredict(self, data, result):
        """Functionality after prediction is computed
        """
        self.predictions = result


    def _predict(self, data):
        """Actual prediction
        """
        raise NotImplementedError


    def predict(self, data):
        """Predict classifier on data

        Shouldn't be overriden in subclasses unless explicitely needed
        to do so. Also subclasses trying to call super class's predict
        should call _predict if within _predict instead of predict()
        since otherwise it would loop
        """
        data = N.array(data)
        if __debug__:
            debug("CLF", "Predicting classifier %s on data %s" \
                % (`self`, `data.shape`))
            tb = traceback.extract_stack(limit=5)
            debug("CLF_TB", "Traceback: %s" % tb)

        self._prepredict(data)
        result = self._predict(data)
        self._postpredict(data, result)
        return result

    def isTrained(self, dataset=None):
        """Either classifier was already trained.

        MUST BE USED WITH CARE IF EVER"""
        if dataset is None:
            # simply return if it was trained on anything
            return not self.__trainednfeatures is None
        else:
            return (self.__trainednfeatures == dataset.nfeatures) \
                   and (self.__trainedid == dataset._id)

    @property
    def trained(self):
        """Either classifier was already trained"""
        return self.isTrained()

    def untrain(self):
        """Reset trained state"""
        self.__trainednfeatures = None


    def _setTrain2predict(self, v):
        """Set the flag for necessary training prior doing prediction

        NOTE: Is not supposed to be called by the user but just by
        derived classes"""
        self.__train2predict = v


    @property
    def train2predict(self):
        """Either classifier has to be trained to predict"""
        return self.__train2predict



#
# Base classifiers of various kinds
#

class BoostedClassifier(Classifier):
    """Classifier containing the farm of other classifiers.

    Should rarely be used directly. Use one of its childs instead
    """

    # should not be needed if we have prediction_values upstairs
    raw_predictions = StateVariable(enabled=False,
        doc="Predictions obtained from each classifier")

    raw_values = StateVariable(enabled=False,
        doc="Values obtained from each classifier")


    def __init__(self, clfs=[], **kwargs):
        """Initialize the instance.

        :Parameters:
          `clfs` : list
            list of classifier instances to use
          kwargs : dict
            dict of keyworded arguments which might get used
            by State or Classifier
        """
        Classifier.__init__(self, **kwargs)

        self.__clfs = None
        """Pylint friendly definition of __clfs"""

        self._setClassifiers(clfs)
        """Store the list of classifiers"""


    def __repr__(self):
        return "<%s(%d classifiers)>" \
               % (self.__class__.__name__, len(self.clfs))


    def _train(self, dataset):
        """Train `BoostedClassifier`
        """
        for clf in self.__clfs:
            clf.train(dataset)


    def _predict(self, data):
        """Predict using `BoostedClassifier`
        """
        raw_predictions = [ clf.predict(data) for clf in self.__clfs ]
        self.raw_predictions = raw_predictions
        assert(len(self.__clfs)>0)
        if self.states.isEnabled("values"):
            # XXX pylint complains that numpy has no array member... weird
            if N.array([x.states.isEnabled("values")
                        for x in self.__clfs]).all():
                values = [ clf.values for clf in self.__clfs ]
                self.raw_values = values
            else:
                warning("One or more classifiers in %s has no 'values' state" %
                        `self` + "enabled, thus BoostedClassifier can't have" +
                        " 'raw_values' state variable defined")

        return raw_predictions


    def _setClassifiers(self, clfs):
        """Set the classifiers used by the boosted classifier

        We have to allow to set list of classifiers after the object
        was actually created. It will be used by
        BoostedMulticlassClassifier
        """
        self.__clfs = clfs
        """Classifiers to use"""

        train2predicts = [clf.train2predict for clf in self.__clfs]
        train2predict = reduce(lambda x, y: x or y, train2predicts, False)
        if __debug__:
            debug("CLFBST", "Setting train2predict=%s for classifiers " \
                   "%s with %s" \
                   % (str(train2predict), `self.__clfs`, str(train2predicts)))
        # set flag if it needs to be trained before predicting
        self._setTrain2predict(train2predict)

    def untrain(self):
        """Untrain `BoostedClassifier`

        Has to untrain any known classifier
        """
        if not self.trained:
            return
        for clf in self.clfs:
            clf.untrain()
        super(BoostedClassifier, self).untrain()

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
        Classifier.__init__(self, train2predict=clf.train2predict, **kwargs)

        self.__clf = clf
        """Store the classifier to use."""


    def _train(self, dataset):
        """Train `ProxyClassifier`
        """
        # base class does nothing much -- just proxies requests to underlying
        # classifier
        self.__clf.train(dataset)

        # for the ease of access
        self.states._copy_states_(self.__clf, deep=False)


    def _predict(self, data):
        """Predict using `ProxyClassifier`
        """
        result = self.__clf.predict(data)
        # for the ease of access
        self.states._copy_states_(self.__clf, deep=False)
        return result


    def untrain(self):
        """Untrain main classifier
        """
        self.clf.untrain()
        super(ProxyClassifier, self).untrain()


    clf = property(lambda x:x.__clf, doc="Used `Classifier`")



#
# Various combiners for CombinedClassifier
#

class PredictionsCombiner(Stateful):
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
                    # we might have multiple labels assigned XXX
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
                warning("We got multiple labels %s which have the " % `maxk` +
                        "same maximal vote %d. XXX disambiguate" % maxv)
            predictions.append(maxk[0])

        self.all_label_counts = all_label_counts
        self.predictions = predictions
        return predictions



class ClassifierCombiner(PredictionsCombiner):
    """Provides a decision using training a classifier on predictions/values

    TODO
    """

    predictions = StateVariable(enabled=True,
        doc="Trained predictions")


    def __init__(self, clf, variables=['predictions']):
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

        self.__variables = variables
        """What state variables of the classifiers to use"""


    def __call__(self, clfs, dataset):
        """
        """
        if len(clfs)==0:
            return []                   # to don't even bother

        # XXX What is it, Exception or Return?
        raise NotImplementedError
        self.predictions = predictions
        return predictions



class CombinedClassifier(BoostedClassifier):
    """`BoostedClassifier` which combines predictions using some `PredictionsCombiner`
    functor.
    """

    def __init__(self, clfs=[], combiner=MaximalVote(), **kwargs):
        """Initialize the instance.

        :Parameters:
          clfs : list of Classifier
            list of classifier instances to use
          combiner : PredictionsCombiner
            callable which takes care about combining multiple
            results into a single one (e.g. maximal vote)
          kwargs : dict
            dict of keyworded arguments which might get used
            by State or Classifier

        NB: `combiner` might need to operate not on 'predictions' descrete
            labels but rather on raw 'class' values classifiers
            estimate (which is pretty much what is stored under
            `values`
        """
        BoostedClassifier.__init__(self, clfs, **kwargs)

        self.__combiner = combiner
        """Functor destined to combine results of multiple classifiers"""


    def __repr__(self):
        return "<%s(%d classifiers, combiner %s)>" \
               % (self.__class__.__name__, len(self.clfs), `self.__combiner`)


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
                    warning("Boosted classifier %s has 'values' state" % `self` +
                            " enabled, but combiner has it active, thus no" +
                            " values could be provided directly, access .clfs")
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

        if len(self.__poslabels)>1:
            self.__predictpos = self.__poslabels
        else:
            self.__predictpos = self.__poslabels[0]

        if len(self.__neglabels)>1:
            self.__predictneg = self.__neglabels
        else:
            self.__predictneg = self.__neglabels[0]


    def __str__(self):
        return "BinaryClassifier +1: %s -1: %s" % (
            `self.__poslabels`, `self.__neglabels`)


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
        self.__clf = clf
        """Store sample instance of basic classifier"""

        # XXX such logic below might go under train....
        if bclf_type == "1-vs-1":
            pass
        elif bclf_type == "1-vs-all":
            raise NotImplementedError
        else:
            raise ValueError, \
                  "Unknown type of classifier %s for " % bclf_type + \
                  "BoostedMulticlassClassifier"
        self.__bclf_type = bclf_type


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
                    clf = _deepcopyclf(self.__clf)
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

    TODO: SplitClassifier and MulticlassClassifier have too much in
          common -- need to refactor: just need a splitter which would
          split dataset in pairs of class labels. MulticlassClassifier
          does just a tiny bit more which might be not necessary at
          all: map sets of labels into 2 categories...
    """

    training_confusions = StateVariable(enabled=True,
        doc="Resultant confusion matrices whenever classifier trained " +
            "on each was tested on 2nd part of the split")

    def __init__(self, clf, splitter=NFoldSplitter(cvtype=1), **kwargs):
        """Initialize the instance

        :Parameters:
          clf : Classifier
            classifier based on which multiple classifiers are created
            for multiclass
          splitter : Splitter
            `Splitter` to use to split the dataset prior training
          """
        CombinedClassifier.__init__(self, **kwargs)
        self.__clf = clf
        """Store sample instance of basic classifier"""
        self.__splitter = splitter


    def _train(self, dataset):
        """Train `SplitClassifier`
        """
        # generate pairs and corresponding classifiers
        bclfs = []
        self.training_confusions = ConfusionMatrix(labels=dataset.uniquelabels)

        # for proper and easier debugging - first define classifiers and then
        # train them
        for split in self.__splitter(dataset):
            if __debug__:
                debug("CLFSPL",
                      "Deepcopying %s for %s" % (`self.__clf`, `self`))
            clf = _deepcopyclf(self.__clf)
            bclfs.append(clf)
        self.clfs = bclfs

        i = 0
        for split in self.__splitter(dataset):
            if __debug__:
                debug("CLFSPL", "Training classifier for split %d" % (i))

            clf = self.clfs[i]
            clf.train(split[0])
            if self.states.isEnabled("training_confusions"):
                predictions = clf.predict(split[1].samples)
                self.training_confusions.add(split[1].labels, predictions)
            i += 1



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
        # for train() we have to provide dataset -- not just samples to train!
        wdataset = dataset.applyMapper(featuresmapper = self.__mapper)
        ProxyClassifier.train(self, wdataset)

    def _predict(self, data):
        """Predict using `MappedClassifier`
        """
        return ProxyClassifier._predict(self, self.__mapper.forward(data))


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


    def _train(self, dataset):
        """Train `FeatureSelectionClassifier`
        """
        # temporarily enable selected_ids
        self.__feature_selection.states._changeTemporarily(
            enable_states=["selected_ids"])

        (wdataset, tdataset) = self.__feature_selection(dataset,
                                                        self.__testdataset)
        if __debug__:
            debug("CLFFS", "{%s} selected %d out of %d features" %
                  (`self.__feature_selection`, wdataset.nfeatures,
                   dataset.nfeatures))

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
        self.states._copy_states_(self.__maskclf, deep=False)


    def _predict(self, data):
        """Predict using `FeatureSelectionClassifier`
        """
        result = self.__maskclf._predict(data)
        # for the ease of access
        self.states._copy_states_(self.__maskclf, deep=False)
        return result

    # XXX Shouldn't that be mappedclf ?
    # YYY yoh: not sure... by nature it is mappedclf, by purpouse it
    # is maskclf using MaskMapper
    maskclf = property(lambda x:x.__maskclf, doc="Used `MappedClassifier`")
    feature_selection = property(lambda x:x.__feature_selection,
                                 doc="Used `FeatureSelection`")

