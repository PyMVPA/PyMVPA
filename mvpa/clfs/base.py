# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Base class for all XXX learners: classifiers and regressions.
"""

__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.support.copy import deepcopy

import time

from mvpa.base.types import is_datasetlike, accepts_dataset_as_samples

from mvpa.datasets.base import dataset
from mvpa.misc.support import idhash
from mvpa.misc.state import StateVariable, ClassWithCollections
from mvpa.misc.param import Parameter
from mvpa.misc.attrmap import AttributeMap

from mvpa.clfs.transerror import ConfusionMatrix, RegressionStatistics

from mvpa.base import warning

if __debug__:
    from mvpa.base import debug

__all__ = [ 'Classifier',
            'accepts_dataset_as_samples', 'accepts_samples_as_dataset',
            'DegenerateInputError', 'FailedToTrainError', 'FailedToPredictError']

def accepts_samples_as_dataset(fx):
    """Decorator to wrap samples into a Dataset.

    Little helper to allow methods to accept plain data whenever
    dataset is generally required.
    """
    def wrap_samples(obj, data, *args, **kwargs):
        if is_datasetlike(data):
            return fx(obj, data, *args, **kwargs)
        else:
            return fx(obj, dataset(data), *args, **kwargs)
    return wrap_samples

class DegenerateInputError(Exception):
    """Exception to be thrown by learners if input data is bogus, i.e. no
    features or samples"""
    pass

class FailedToTrainError(Exception):
    """Exception to be thrown whenever classifier fails to learn for
    some reason"""
    pass

class FailedToPredictError(Exception):
    """Exception to be thrown whenever classifier fails to provide predictions.
    Usually happens if it was trained on degenerate data but without any complaints.
    """
    pass


class Classifier(ClassWithCollections):
    """Abstract classifier class to be inherited by all classifiers
    """

    # Kept separate from doc to don't pollute help(clf), especially if
    # we including help for the parent class
    _DEV__doc__ = """
    Required behavior:

    For every classifier is has to be possible to be instantiated without
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
             respective class. However the semantics might differ between
             classifiers, e.g. kNN would probably store distances to class-
             neighbors, where PLR would store the raw function value of the
             logistic function. So in the case of kNN low is predictive and for
             PLR high is predictive. Don't know if there is the need to unify
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
    # Additionally Michael thinks ATM that additional information might be
    # necessary in some situations (e.g. reasonably predefined parameter range,
    # minimal iteration stepsize, ...), therefore the value to each key should
    # also be a dict or we should use mvpa.misc.param.Parameter'...

    trained_labels = StateVariable(enabled=True,
        doc="Set of unique labels it has been trained on")

    trained_nsamples = StateVariable(enabled=True,
        doc="Number of samples it has been trained on")

    trained_dataset = StateVariable(enabled=False,
        doc="The dataset it has been trained on")

    training_confusion = StateVariable(enabled=False,
        doc="Confusion matrix of learning performance")

    predictions = StateVariable(enabled=True,
        doc="Most recent set of predictions")

    values = StateVariable(enabled=True,
        doc="Internal classifier values the most recent " +
            "predictions are based on")

    training_time = StateVariable(enabled=True,
        doc="Time (in seconds) which took classifier to train")

    predicting_time = StateVariable(enabled=True,
        doc="Time (in seconds) which took classifier to predict")

    feature_ids = StateVariable(enabled=False,
        doc="Feature IDS which were used for the actual training.")

    __tags__ = []
    """Describes some specifics about the classifier -- is that it is
    doing regression for instance...."""

    # TODO: make it available only for actually retrainable classifiers
    retrainable = Parameter(False, allowedtype='bool',
        doc="""Either to enable retraining for 'retrainable' classifier.""",
        index=1002)


    def __init__(self, **kwargs):
        ClassWithCollections.__init__(self, **kwargs)

        # XXX
        # the place to map literal to numerical labels (and back)
        # this needs to be in the base class, since some classifiers also
        # have this nasty 'regression' mode, and the code in this class
        # needs to deal with converting the regression output into discrete
        # labels
        # however, preferably the mapping should be kept in the respective
        # low-level implementations that need it
        self._attrmap = AttributeMap()


        self.__trainednfeatures = None
        """Stores number of features for which classifier was trained.
        If None -- it wasn't trained at all"""

        self._setRetrainable(self.params.retrainable, force=True)

        # deprecate
        #self.__trainedidhash = None
        #"""Stores id of the dataset on which it was trained to signal
        #in trained() if it was trained already on the same dataset"""

    @property
    def __summary_class__(self):
        if 'regression' in self.__tags__:
            return RegressionStatistics
        else:
            return ConfusionMatrix

    @property
    def __is_regression__(self):
        return 'regression' in self.__tags__

    def __str__(self):
        if __debug__ and 'CLF_' in debug.active:
            return "%s / %s" % (repr(self), super(Classifier, self).__str__())
        else:
            return repr(self)

    def __repr__(self, prefixes=[]):
        return super(Classifier, self).__repr__(prefixes=prefixes)


    def _pretrain(self, dataset):
        """Functionality prior to training
        """
        # So we reset all state variables and may be free up some memory
        # explicitly
        params = self.params
        if not params.retrainable:
            self.untrain()
        else:
            # just reset the states, do not untrain
            self.states.reset()
            if not self.__changedData_isset:
                self.__resetChangedData()
                _changedData = self._changedData
                __idhashes = self.__idhashes
                __invalidatedChangedData = self.__invalidatedChangedData

                # if we don't know what was changed we need to figure
                # them out
                if __debug__:
                    debug('CLF_', "IDHashes are %s" % (__idhashes))

                # Look at the data if any was changed
                for key, data_ in (('traindata', dataset.samples),
                                   ('labels', dataset.labels)):
                    _changedData[key] = self.__wasDataChanged(key, data_)
                    # if those idhashes were invalidated by retraining
                    # we need to adjust _changedData accordingly
                    if __invalidatedChangedData.get(key, False):
                        if __debug__ and not _changedData[key]:
                            debug('CLF_', 'Found that idhash for %s was '
                                  'invalidated by retraining' % key)
                        _changedData[key] = True

                # Look at the parameters
                for col in self._paramscols:
                    changedParams = self._collections[col].which_set()
                    if len(changedParams):
                        _changedData[col] = changedParams

                self.__invalidatedChangedData = {} # reset it on training

                if __debug__:
                    debug('CLF_', "Obtained _changedData is %s"
                          % (self._changedData))


    def _posttrain(self, dataset):
        """Functionality post training

        For instance -- computing confusion matrix.

        Parameters
        ----------
        dataset : Dataset
          Data which was used for training
        """
        states = self.states
        if states.is_enabled('trained_labels'):
            states.trained_labels = dataset.sa['labels'].unique

        states.trained_dataset = dataset
        states.trained_nsamples = dataset.nsamples

        # needs to be assigned first since below we use predict
        self.__trainednfeatures = dataset.nfeatures

        if __debug__ and 'CHECK_TRAINED' in debug.active:
            self.__trainedidhash = dataset.idhash

        if self.states.is_enabled('training_confusion') and \
               not self.states.is_set('training_confusion'):
            # we should not store predictions for training data,
            # it is confusing imho (yoh)
            self.states.change_temporarily(
                disable_states=["predictions"])
            if self.params.retrainable:
                # we would need to recheck if data is the same,
                # XXX think if there is a way to make this all
                # efficient. For now, probably, retrainable
                # classifiers have no chance but not to use
                # training_confusion... sad
                self.__changedData_isset = False
            predictions = self.predict(dataset)
            self.states.reset_changed_temporarily()
            self.states.training_confusion = self.__summary_class__(
                targets=dataset.sa.labels,
                predictions=predictions)

        if self.states.is_enabled('feature_ids'):
            self.states.feature_ids = self._getFeatureIds()


    def _getFeatureIds(self):
        """Virtual method to return feature_ids used while training

        Is not intended to be called anywhere but from _posttrain,
        thus classifier is assumed to be trained at this point
        """
        # By default all features are used
        return range(self.__trainednfeatures)


    def summary(self):
        """Providing summary over the classifier"""

        s = "Classifier %s" % self
        states = self.states
        states_enabled = states.enabled

        if self.trained:
            s += "\n trained"
            if states.is_set('training_time'):
                s += ' in %.3g sec' % states.training_time
            s += ' on data with'
            if states.is_set('trained_labels'):
                s += ' labels:%s' % list(states.trained_labels)

            nsamples, nchunks = None, None
            if states.is_set('trained_nsamples'):
                nsamples = states.trained_nsamples
            if states.is_set('trained_dataset'):
                td = states.trained_dataset
                nsamples, nchunks = td.nsamples, len(td.sa['chunks'].unique)
            if nsamples is not None:
                s += ' #samples:%d' % nsamples
            if nchunks is not None:
                s += ' #chunks:%d' % nchunks

            s += " #features:%d" % self.__trainednfeatures
            if states.is_set('feature_ids'):
                s += ", used #features:%d" % len(states.feature_ids)
            if states.is_set('training_confusion'):
                s += ", training error:%.3g" % states.training_confusion.error
        else:
            s += "\n not yet trained"

        if len(states_enabled):
            s += "\n enabled states:%s" % ', '.join([str(states[x])
                                                     for x in states_enabled])
        return s


    def clone(self):
        """Create full copy of the classifier.

        It might require classifier to be untrained first due to
        present SWIG bindings.

        TODO: think about proper re-implementation, without enrollment of deepcopy
        """
        if __debug__:
            debug("CLF", "Cloning %s#%s" % (self, id(self)))
        try:
            return deepcopy(self)
        except:
            self.untrain()
            return deepcopy(self)


    def _train(self, dataset):
        """Function to be actually overridden in derived classes
        """
        raise NotImplementedError


    def train(self, dataset):
        """Train classifier on a dataset

        Shouldn't be overridden in subclasses unless explicitly needed
        to do so
        """
        if dataset.nfeatures == 0 or dataset.nsamples == 0:
            raise DegenerateInputError, \
                  "Cannot train classifier on degenerate data %s" % dataset
        if __debug__:
            debug("CLF", "Training classifier %(clf)s on dataset %(dataset)s",
                  msgargs={'clf':self, 'dataset':dataset})

        self._pretrain(dataset)

        # remember the time when started training
        t0 = time.time()

        if dataset.nfeatures > 0:

            result = self._train(dataset)
        else:
            warning("Trying to train on dataset with no features present")
            if __debug__:
                debug("CLF",
                      "No features present for training, no actual training " \
                      "is called")
            result = None

        self.states.training_time = time.time() - t0
        self._posttrain(dataset)
        return result


    def _prepredict(self, dataset):
        """Functionality prior prediction
        """
        if not ('notrain2predict' in self.__tags__):
            # check if classifier was trained if that is needed
            if not self.trained:
                raise ValueError, \
                      "Classifier %s wasn't yet trained, therefore can't " \
                      "predict" % self
            nfeatures = dataset.nfeatures #data.shape[1]
            # check if number of features is the same as in the data
            # it was trained on
            if nfeatures != self.__trainednfeatures:
                raise ValueError, \
                      "Classifier %s was trained on data with %d features, " % \
                      (self, self.__trainednfeatures) + \
                      "thus can't predict for %d features" % nfeatures


        if self.params.retrainable:
            if not self.__changedData_isset:
                self.__resetChangedData()
                _changedData = self._changedData
                data = N.asanyarray(dataset.samples)
                _changedData['testdata'] = \
                                        self.__wasDataChanged('testdata', data)
                if __debug__:
                    debug('CLF_', "prepredict: Obtained _changedData is %s"
                          % (_changedData))


    def _postpredict(self, dataset, result):
        """Functionality after prediction is computed
        """
        self.states.predictions = result
        if self.params.retrainable:
            self.__changedData_isset = False

    def _predict(self, dataset):
        """Actual prediction
        """
        raise NotImplementedError

    @accepts_samples_as_dataset
    def predict(self, dataset):
        """Predict classifier on data

        Shouldn't be overridden in subclasses unless explicitly needed
        to do so. Also subclasses trying to call super class's predict
        should call _predict if within _predict instead of predict()
        since otherwise it would loop
        """
        ## ??? yoh: changed to asany from as without exhaustive check
        data = N.asanyarray(dataset.samples)
        if __debug__:
            debug("CLF", "Predicting classifier %(clf)s on ds %(dataset)s",
                msgargs={'clf':self, 'dataset':dataset})

        # remember the time when started computing predictions
        t0 = time.time()

        states = self.states
        # to assure that those are reset (could be set due to testing
        # post-training)
        states.reset(['values', 'predictions'])

        self._prepredict(dataset)

        if self.__trainednfeatures > 0 \
               or 'notrain2predict' in self.__tags__:
            result = self._predict(dataset)
        else:
            warning("Trying to predict using classifier trained on no features")
            if __debug__:
                debug("CLF",
                      "No features were present for training, prediction is " \
                      "bogus")
            result = [None]*data.shape[0]

        states.predicting_time = time.time() - t0

        # with a labels mapping in-place, we also need to go back to the
        # literal labels
        if self._attrmap:
            try:
                result = self._attrmap.to_literal(result)
            except KeyError, e:
                raise FailedToPredictError, \
                      "Failed to convert predictions from numeric into " \
                      "literals: %s" % e

        self._postpredict(dataset, result)
        return result

    # XXX deprecate ???
    def isTrained(self, dataset=None):
        """Either classifier was already trained.

        MUST BE USED WITH CARE IF EVER"""
        if dataset is None:
            # simply return if it was trained on anything
            return not self.__trainednfeatures is None
        else:
            res = (self.__trainednfeatures == dataset.nfeatures)
            if __debug__ and 'CHECK_TRAINED' in debug.active:
                res2 = (self.__trainedidhash == dataset.idhash)
                if res2 != res:
                    raise RuntimeError, \
                          "isTrained is weak and shouldn't be relied upon. " \
                          "Got result %b although comparing of idhash says %b" \
                          % (res, res2)
            return res


    @property
    def trained(self):
        """Either classifier was already trained"""
        return self.isTrained()

    def untrain(self):
        """Reset trained state"""
        # any previous apping is obsolete now
        self._attrmap.clear()

        self.__trainednfeatures = None
        # probably not needed... retrainable shouldn't be fully untrained
        # or should be???
        #if self.params.retrainable:
        #    # ??? don't duplicate the code ;-)
        #    self.__idhashes = {'traindata': None, 'labels': None,
        #                       'testdata': None, 'testtraindata': None}
        super(Classifier, self).reset()


    def getSensitivityAnalyzer(self, **kwargs):
        """Factory method to return an appropriate sensitivity analyzer for
        the respective classifier."""
        raise NotImplementedError


    #
    # Methods which are needed for retrainable classifiers
    #
    def _setRetrainable(self, value, force=False):
        """Assign value of retrainable parameter

        If retrainable flag is to be changed, classifier has to be
        untrained.  Also internal attributes such as _changedData,
        __changedData_isset, and __idhashes should be initialized if
        it becomes retrainable
        """
        pretrainable = self.params['retrainable']
        if (force or value != pretrainable.value) \
               and 'retrainable' in self.__tags__:
            if __debug__:
                debug("CLF_", "Setting retrainable to %s" % value)
            if 'meta' in self.__tags__:
                warning("Retrainability is not yet crafted/tested for "
                        "meta classifiers. Unpredictable behavior might occur")
            # assure that we don't drag anything behind
            if self.trained:
                self.untrain()
            states = self.states
            if not value and states.has_key('retrained'):
                states.remove('retrained')
                states.remove('repredicted')
            if value:
                if not 'retrainable' in self.__tags__:
                    warning("Setting of flag retrainable for %s has no effect"
                            " since classifier has no such capability. It would"
                            " just lead to resources consumption and slowdown"
                            % self)
                states['retrained'] = StateVariable(enabled=True,
                        doc="Either retrainable classifier was retrained")
                states['repredicted'] = StateVariable(enabled=True,
                        doc="Either retrainable classifier was repredicted")

            pretrainable.value = value

            # if retrainable we need to keep track of things
            if value:
                self.__idhashes = {'traindata': None, 'labels': None,
                                   'testdata': None} #, 'testtraindata': None}
                if __debug__ and 'CHECK_RETRAIN' in debug.active:
                    # ??? it is not clear though if idhash is faster than
                    # simple comparison of (dataset != __traineddataset).any(),
                    # but if we like to get rid of __traineddataset then we
                    # should use idhash anyways
                    self.__trained = self.__idhashes.copy() # just same Nones
                self.__resetChangedData()
                self.__invalidatedChangedData = {}
            elif 'retrainable' in self.__tags__:
                #self.__resetChangedData()
                self.__changedData_isset = False
                self._changedData = None
                self.__idhashes = None
                if __debug__ and 'CHECK_RETRAIN' in debug.active:
                    self.__trained = None

    def __resetChangedData(self):
        """For retrainable classifier we keep track of what was changed
        This function resets that dictionary
        """
        if __debug__:
            debug('CLF_',
                  'Retrainable: resetting flags on either data was changed')
        keys = self.__idhashes.keys() + self._paramscols
        # we might like to just reinit values to False???
        #_changedData = self._changedData
        #if isinstance(_changedData, dict):
        #    for key in _changedData.keys():
        #        _changedData[key] = False
        self._changedData = dict(zip(keys, [False]*len(keys)))
        self.__changedData_isset = False


    def __wasDataChanged(self, key, entry, update=True):
        """Check if given entry was changed from what known prior.

        If so -- store only the ones needed for retrainable beastie
        """
        idhash_ = idhash(entry)
        __idhashes = self.__idhashes

        changed = __idhashes[key] != idhash_
        if __debug__ and 'CHECK_RETRAIN' in debug.active:
            __trained = self.__trained
            changed2 = entry != __trained[key]
            if isinstance(changed2, N.ndarray):
                changed2 = changed2.any()
            if changed != changed2 and not changed:
                raise RuntimeError, \
                  'idhash found to be weak for %s. Though hashid %s!=%s %s, '\
                  'values %s!=%s %s' % \
                  (key, idhash_, __idhashes[key], changed,
                   entry, __trained[key], changed2)
            if update:
                __trained[key] = entry

        if __debug__ and changed:
            debug('CLF_', "Changed %s from %s to %s.%s"
                      % (key, __idhashes[key], idhash_,
                         ('','updated')[int(update)]))
        if update:
            __idhashes[key] = idhash_

        return changed


    # def __updateHashIds(self, key, data):
    #     """Is twofold operation: updates hashid if was said that it changed.
    #
    #     or if it wasn't said that data changed, but CHECK_RETRAIN and it found
    #     to be changed -- raise Exception
    #     """
    #
    #     check_retrain = __debug__ and 'CHECK_RETRAIN' in debug.active
    #     chd = self._changedData
    #
    #     # we need to updated idhashes
    #     if chd[key] or check_retrain:
    #         keychanged = self.__wasDataChanged(key, data)
    #     if check_retrain and keychanged and not chd[key]:
    #         raise RuntimeError, \
    #               "Data %s found changed although wasn't " \
    #               "labeled as such" % key


    #
    # Additional API which is specific only for retrainable classifiers.
    # For now it would just puke if asked from not retrainable one.
    #
    # Might come useful and efficient for statistics testing, so if just
    # labels of dataset changed, then
    #  self.retrain(dataset, labels=True)
    # would cause efficient retraining (no kernels recomputed etc)
    # and subsequent self.repredict(data) should be also quite fase ;-)

    def retrain(self, dataset, **kwargs):
        """Helper to avoid check if data was changed actually changed

        Useful if just some aspects of classifier were changed since
        its previous training. For instance if dataset wasn't changed
        but only classifier parameters, then kernel matrix does not
        have to be computed.

        Words of caution: classifier must be previously trained,
        results always should first be compared to the results on not
        'retrainable' classifier (without calling retrain). Some
        additional checks are enabled if debug id 'CHECK_RETRAIN' is
        enabled, to guard against obvious mistakes.

        Parameters
        ----------
        kwargs
          that is what _changedData gets updated with. So, smth like
          `(params=['C'], labels=True)` if parameter C and labels
          got changed
        """
        # Note that it also demolishes anything for repredicting,
        # which should be ok in most of the cases
        if __debug__:
            if not self.params.retrainable:
                raise RuntimeError, \
                      "Do not use re(train,predict) on non-retrainable %s" % \
                      self

            if kwargs.has_key('params') or kwargs.has_key('kernel_params'):
                raise ValueError, \
                      "Retraining for changed params not working yet"

        self.__resetChangedData()

        # local bindings
        chd = self._changedData
        ichd = self.__invalidatedChangedData

        chd.update(kwargs)
        # mark for future 'train()' items which are explicitely
        # mentioned as changed
        for key, value in kwargs.iteritems():
            if value:
                ichd[key] = True
        self.__changedData_isset = True

        # To check if we are not fooled
        if __debug__ and 'CHECK_RETRAIN' in debug.active:
            for key, data_ in (('traindata', dataset.samples),
                               ('labels', dataset.labels)):
                # so it wasn't told to be invalid
                if not chd[key] and not ichd.get(key, False):
                    if self.__wasDataChanged(key, data_, update=False):
                        raise RuntimeError, \
                              "Data %s found changed although wasn't " \
                              "labeled as such" % key

        # TODO: parameters of classifiers... for now there is explicit
        # 'forbidance' above

        # Below check should be superseeded by check above, thus never occur.
        # remove later on ???
        if __debug__ and 'CHECK_RETRAIN' in debug.active and self.trained \
               and not self._changedData['traindata'] \
               and self.__trained['traindata'].shape != dataset.samples.shape:
            raise ValueError, "In retrain got dataset with %s size, " \
                  "whenever previousely was trained on %s size" \
                  % (dataset.samples.shape, self.__trained['traindata'].shape)
        self.train(dataset)


    @accepts_samples_as_dataset
    def repredict(self, dataset, **kwargs):
        """Helper to avoid check if data was changed actually changed

        Useful if classifier was (re)trained but with the same data
        (so just parameters were changed), so that it could be
        repredicted easily (on the same data as before) without
        recomputing for instance train/test kernel matrix. Should be
        used with caution and always compared to the results on not
        'retrainable' classifier. Some additional checks are enabled
        if debug id 'CHECK_RETRAIN' is enabled, to guard against
        obvious mistakes.

        Parameters
        ----------
        dataset
          dataset which is conventionally given to predict
        kwargs
          that is what _changedData gets updated with. So, smth like
          `(params=['C'], labels=True)` if parameter C and labels
          got changed
        """
        if len(kwargs)>0:
            raise RuntimeError, \
                  "repredict for now should be used without params since " \
                  "it makes little sense to repredict if anything got changed"
        if __debug__ and not self.params.retrainable:
            raise RuntimeError, \
                  "Do not use retrain/repredict on non-retrainable classifiers"

        self.__resetChangedData()
        chd = self._changedData
        chd.update(**kwargs)
        self.__changedData_isset = True


        # check if we are attempted to perform on the same data
        if __debug__ and 'CHECK_RETRAIN' in debug.active:
            for key, data_ in (('testdata', dataset.samples),):
                # so it wasn't told to be invalid
                #if not chd[key]:# and not ichd.get(key, False):
                if self.__wasDataChanged(key, data_, update=False):
                    raise RuntimeError, \
                          "Data %s found changed although wasn't " \
                          "labeled as such" % key

        # Should be superseded by above
        # remove in future???
        if __debug__ and 'CHECK_RETRAIN' in debug.active \
               and not self._changedData['testdata'] \
               and self.__trained['testdata'].shape != dataset.samples.shape:
            raise ValueError, "In repredict got dataset with %s size, " \
                  "whenever previously was trained on %s size" \
                  % (dataset.samples.shape, self.__trained['testdata'].shape)

        return self.predict(dataset)


    # TODO: callback into retrainable parameter
    #retrainable = property(fget=_getRetrainable, fset=_setRetrainable,
    #                  doc="Specifies either classifier should be retrainable")
