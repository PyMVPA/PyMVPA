#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: Recursive feature elimination"""

import numpy as N

from mvpa.misc.support import buildConfusionMatrix
from mvpa.misc.exceptions import UnknownStateError
from mvpa.misc.vproperty import VProperty

class FeatureSelection(object):
    """ Base class for any feature selection

    TODO...
    """

    def __init__(self):
        self.__mask = None
        """Binary mask defining the voxels which were selected"""


    def __call__(self, dataset, callables=[]):
        """Invocation of the feature selection

        - `dataset`: actually dataset.
        - `callables`: a list of functors to be called with locals()

        Returns a dataset with selected features.  Derived classes
        must provide interface to access other relevant to the feature
        selection process information (e.g. mask, elimination step
        (in RFE), etc)
        """
        raise NotImplementedError


    def getMask(self):
        """ Returns a mask computed during previous call()
        """
        if self.__mask is None:
            raise UnknownStateError
        return self.__mask

    mask = VProperty(fget=getMask)


#
# TODO: Elaborate this sceletons whenever it becomes necessary
#

class StoppingCriterion(object):
    """Base class for all functors to decide when to stop RFE (or may
    be general optimization... so it probably will be moved out into
    some other module
    """

    def __call__(self, errors):
        """Instruct when to stop

        Returns tuple (`go`, `isthebest`)
        """
        raise NotImplementedError


class StopNBackHistoryCriterion(StoppingCriterion):
    """ Stop computation if for a number of steps error was increasing
    """

    def __init__(self, steps=10, func=min):
        """Initialize with number of steps
        `steps`: int, for how many steps to check after optimal value
        `fun`: functor, to select the best results. Defaults to min
        """
        if steps<0:
            raise ValueError, \
                  "Number of steps (got %d) should be non-negative" % steps
        self.__steps = steps
        self.__func = func


    def __call__(self, errors):
        isbest = False
        stop = False

        # just to prevent ValueError
        if len(errors)==0:
            return (isbest, stop)

        minerror = self.__func(errors)
        minindex = errors.index(minerror)

        # if minimal is the last one reported -- it is the best
        if minindex == len(errors)-1:
            isbest = True

        # if number of elements after the min >= len -- stop
        if len(errors) - minindex >= self.__steps:
            stop = True

        return (stop, isbest)

    steps = property(fget=lambda x:x.__steps)

class FeatureSelector(object):
    """Base class to implement functors to select the set of properties
    """
    pass


class XPercentFeatureSelector(FeatureSelector):
    """Given a sensitivity map, provide Ids given a percentage of features

    Since silly Yarik could not recall alternative to "proportion" to select
    in units like 0.05 for 5%, now this selector does take values in percents
    TODO: Should be DiscardSelector on top of it... __repr__, ndiscarded should
          belong to it.
    """

    def __init__(self, perc_discard=5.0, removeminimal=True):
        """XXX???
        """
        if perc_discard>100 or perc_discard<0:
            raise ValueError, \
                  "Percentage (%f) cannot be outside of [0,100]" \
                  % perc_discard
        self.__perc_discard = perc_discard
        self.__removeminimal = removeminimal
        self.__ndiscarded = None
        """Store number of discarded since for a given value we might
        remove more than 1 if they are all equal -- it would be unfair
        to leave some features in while another with the same value
        got discarded
        """


    def __repr__(self):
        s = "%s: perc=%f minimal=%s" % (
            self.__name__, self.__perc_discard, self.__removeminimal)
        if not self.__ndiscarded is None:
            s += " discarded: %d" % self.ndiscarded


    def __call__(self, sensitivity):
        nfeatures = len(sensitivity)
        # how many to discard
        nremove = self.__perc * nfeatures * 1.0 / 100.0
        sensmap = N.array(sensitivity)  # assure that it is ndarray
        sensmap2 = sensmap.copy()       # make a copy to sort
        sensmap2.sort()                 # sort inplace
        if self.__removeminimal:
            good_ids = sensmap[sensmap>sensmap2[nremove-1]]
        else:
            # remove maximal elements
            good_ids = sensmap[sensmap<sensmap2[-nremove]]
        # compute actual number of discarded elements
        self.__ndiscarded = nfeatures - len(good_ids)
        return good_ids


    def getNDiscarded(self):
        """Return number of discarded elements

        Raises an UnknownStateError exception if the instance wasn't
        called yet
        """
        if self.__ndiscarded == None:
            raise UnknownStateError
        return self.__ndiscarded

    ndiscarded = property(fget=getNDiscarded)

# TODO: Abs value of sensitivity should be able to rule RFE
# Often it is what abs value of the sensitivity is what matters.
# So we should either provide a simple decorator around arbitrary
# FeatureSelector to convert sensitivities to abs values before calling
# actual selector, or a decorator around SensitivityEstimators

class RFE(FeatureSelection):
    """ Recursive feature elimination.
    """

    def __init__(self,
                 sensana,
                 feature_selector=XPercentFeatureSelector,
                 stopping_criterion=StopNBackHistoryCriterion,
                 error_oracle=None
                 ):
        """ Initialize recurse feature elimination
        `sensana`: `SensitivityAnalyzer`
        `feature_selector`: functor
        `stopping_criterion`: functor
        """

        self.__sensana = sensana
        """Sensitivity analyzer used to call at each step"""

        self.__feature_selector = feature_selector
        """Functor which takes care about removing some features"""

        self.__stopping_criterion = stopping_criterion

        self.__error_oracle = error_oracle


    def __call__(self, dataset, callables=[]):
        """Proceed and select the features recursively eliminating less
        important ones.
        """
        errors = []
        go = True
        result = None

        while dataset.nfeatures>0:
            # Compute
            sensitivity = self.__sensana(dataset)
            # Record the error
            errors.append(error_oracle(dataset))
            # Check if it is time to stop and if we got
            # the best result
            (go, isthebest) = self.__stopping_criterion(errors)

            # store result
            if isthebest:
                result = dataset

            # stop if it is time to finish
            if dataset.nfeatures == 1 or not go:
                break

            # Select features to preserve
            selected_ids = self.__feature_selector(sensitivity)

            # Create a dataset only with selected features
            newdataset = dataset.selectFeatures(selected_ids)

            for callable_ in callables:
                callable_(locals())

            # reassign, so in callables we got both older and new
            # datasets
            dataset = newdataset

        return result

    def __trainClassifier(self):
        self.__clf.train( self.__pattern )
        self.__testClassifierOnTrainingSet()

        # feature rating: large values == important
        fr = self.__clf.getFeatureBenchmark()


    def __testClassifierOnTrainingSet(self):
        predictions = N.array(self.__clf.predict( self.pattern.samples ))

        self.__training_confusion_mat = \
            buildConfusionMatrix( self.pattern.reglabels,
                                  self.pattern.regs,
                                  predictions )
        self.__training_perf = \
            float(self.trainconfmat.diagonal().sum()) / self.trainconfmat.sum()


    def selectFeatures( self, n, eliminate_by = 'fraction',
                        kill_per_iter = 0.1 ):
        """ Select n predictive features from the dataset.

        This method performs as many iterations as necessary until only 'n'
        feature are remaining. The number of iteration depend on the elimination
        method that is used.

        'eliminate_by' and 'kill_per_iter' are used to determine how many
        features are removed in each iteration.

        Supported values for 'eliminate_by' and the associated meaning of
        'kill_per_iter' are listed below:

        eliminate_by: 'fraction'
            in this case 100*'kill_per_iter' represents the percentage of
            features that are removed at each iteration

        eliminate_by: 'number'
            each iteration removed 'number' features
        """
        # are we talking integers?
        n = int( n )

        # elimination log mask
        elim_mask = N.zeros( self.pattern.mapper.dsshape, dtype='uint' )

        # do it until there are more feature than there should be
        while n < self.pattern.nfeatures:
            # get importance of each remaining feature
            featrank = self.__clf.getFeatureBenchmark()

            # determine the number of features to be eliminated
            if eliminate_by == 'fraction':
                nkill = int( len(featrank) * kill_per_iter )
            elif eliminate_by == 'number':
                nkill = kill_per_iter
            else:
                raise ValueError, 'Unknown elimination method: ' \
                                  + str(eliminate_by)

            # limit number of eliminations if necessary (would have less than
            # n features after elimination)
            if nkill > self.pattern.nfeatures - n:
                nkill = self.pattern.nfeatures - n

            # make sure to remove one feature at least to prevent endless
            # loops
            if nkill < 1:
                nkill = 1

            if self.__verbose:
                print "Removing", nkill, "features.",

            # eliminate the first 'nkill' features (with the lowest ranking)
            self.__pattern = \
                self.pattern.selectFeatures( featrank.argsort()[nkill:] )

            # increment all features still present in the elim mask
            elim_mask[ self.pattern.mapper.getMask(copy=False) ] += 1

            # retrain the classifier with the new feature set
            self.__trainClassifier()

            if self.__verbose:
                print "New performance:", self.trainperf

        return elim_mask


    def killNFeatures( self, n,
                       eliminate_by = 'number',
                       kill_per_iter = 1 ):
        """ Eliminates the 'n' least predictive features.

        Please see the documentation of the selectFeatures() methods for
        information about the remaining arguments.
        """
        self.selectFeatures( self.pattern.nfeatures - n,
                             eliminate_by = eliminate_by,
                             kill_per_iter = kill_per_iter )


    def killFeatureFraction( self, frac,
                             eliminate_by = 'fraction',
                             kill_per_iter = 0.1 ):
        """ Eliminates the 100*frac percent least predictive features.

        Please see the documentation of the selectFeatures() methods for
        information about the remaining arguments.
        """
        nkill = int(frac * self.pattern.nfeatures)
        self.selectFeatures( self.pattern.nfeatures - nkill,
                             eliminate_by = eliminate_by,
                             kill_per_iter = kill_per_iter )


    def testSelection( self, pattern ):
        """ Test the performance of the current feature set on a given dataset.

        The datasets in 'pattern' (MVPAPattern) are passed to the internal
        classifier object. No retraining is performed. The classifiers
        predictions are compared to the regressor values in 'pattern'.

        'pattern' either has to provide the same number of features as the
        current feature set or has to match the original data shape. In case of
        the latter an automatic feature selection is performed and only those
        features from 'pattern' are used for the prediction that are also
        present in the current internal feature set.

        The method returns a 3-tuple:
            ( list of classifier prediction (one per dataset),
              mean classifier performance (0 < x < 1),
              confusion matrix of classifications )
        """
        if not pattern.nfeatures == self.pattern.nfeatures:
            # select necessary features from the provided patterns
            masked = pattern.selectFeaturesByMask( 
                            self.pattern.mapper.getMask() )
        else:
            masked = pattern

        # get the predictions from the classifier
        predicted = self.clf.predict( masked.samples )

        confmat = buildConfusionMatrix( self.pattern.reglabels,
                                        masked.regs,
                                        predicted )

        perf = N.mean( predicted == masked.regs )

        return predicted, perf, confmat


    def setVerbosity( self, value ):
        """ If verbosity is set to True, some status messages will be printed
        during recursive feature selection.
        """
        self.__verbose = value


    # properties
    pattern = property( fget=lambda self: self.__pattern )
    verbosity = property( fget=lambda self: self.__verbose,
                          fset=setVerbosity )
    clf = property( fget=lambda self: self.__clf )
    trainconfmat = property( fget=lambda self: self.__training_confusion_mat )
    trainperf = property( fget=lambda self: self.__training_perf )
