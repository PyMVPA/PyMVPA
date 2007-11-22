#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Recursive feature elimination"""

import numpy as N

from math import floor, ceil

from mvpa.algorithms.featsel import FeatureSelection, \
                                    StopNBackHistoryCriterion, \
                                    XPercentFeatureSelector
from mvpa.misc.support import buildConfusionMatrix
from mvpa.misc.exceptions import UnknownStateError
from mvpa.misc.vproperty import VProperty
from mvpa.misc.state import State

if __debug__:
    from mvpa.misc import debug

# TODO: Abs value of sensitivity should be able to rule RFE
# Often it is what abs value of the sensitivity is what matters.
# So we should either provide a simple decorator around arbitrary
# FeatureSelector to convert sensitivities to abs values before calling
# actual selector, or a decorator around SensitivityEstimators

class RFE(FeatureSelection, State):
    """ Recursive feature elimination.
    """

    def __init__(self,
                 sensitivity_analyzer,
                 transfer_error,
                 feature_selector=XPercentFeatureSelector(),
                 stopping_criterion=StopNBackHistoryCriterion(),
                 train_clf=True
                 ):
        """ Initialize recursive feature elimination

        Parameter
        ---------

        - `sensitivity_analyzer`: `SensitivityAnalyzer`
        - `transfer_error`: `TransferError` instance used to compute the
                transfer error of a classifier based on a certain feature set
                on test dataset.
        - `feature_selector`: Functor. Given a sensitivity map it has to return
                the ids of those features that should be kept.
        - `stopping_criterion`: Functor. Given a list of error values it has
                to return a tuple of two booleans. First values must indicate
                whether the criterion is fulfilled and the second value signals
                whether the latest error values is the total minimum.
        - `train_clf`: Flag whether the classifier in `transfer_error` should
                be trained before computing the error. In general this is
                required, but if the `sensitivity_analyzer` and
                `transfer_error` share and make use of the same classifier and
                can be switched off to save CPU cycles.
        """
        # base init first
        State.__init__(self)

        self.__sensitivity_analyzer = sensitivity_analyzer
        """Sensitivity analyzer used to call at each step."""

        self.__transfer_error = transfer_error
        """Compute transfer error for each feature set."""

        self.__feature_selector = feature_selector
        """Functor which takes care about removing some features."""

        self.__stopping_criterion = stopping_criterion

        self.__train_clf = train_clf
        """Flag whether training classifier is required."""

        # register some
        self._registerState("errors")


        """
    error_oracle = lamda x:errofx(clf.predict(
        x.mapper(testdata.mapper.reverse(testdata))))

    clf = SVM()
    sensitivity_analyzer = linearSVMSensitivity(clf)
    error_oracle = lambda x,y:errorfx(clf.predict(y))
    ClassifierBasedSensitivity(Classifier, SensitivityAnalyzer)

    error_oracle = lambda x,y: classifierBasedSensitivity.predict_error(y)
    sensitivity_analyzer = classifierBasedSensitivity

    sensitivity_analyzer = GLMSensitivity(...)
    def train_and_predict_error(clf)
    error_oracle =
    rfe = RFE(..., error_oracle, sensitivity_analyzer)
    """

    def __call__(self, dataset, testdataset, callables=[]):
        """Proceed and select the features recursively eliminating less
        important ones.
        """
        errors = []
        stop = False
        result = None
        newtestdataset = None
        step = 0
        while dataset.nfeatures>0:
            # Compute
            sensitivity = self.__sensitivity_analyzer(dataset)

            # do not retrain clf if not necessary
            if self.__train_clf:
                error = self.__transfer_error(testdataset, dataset)
            else:
                error = self.__transfer_error(testdataset, None)

            # Record the error
            errors.append(error)

            # Check if it is time to stop and if we got
            # the best result
            (stop, isthebest) = self.__stopping_criterion(errors)

            if __debug__:
                debug('RFEC',
                      "Step %d: nfeatures=%d error=%.4f best/stop=%d/%d" %
                      (step, dataset.nfeatures, errors[-1], isthebest, stop))

            # store result
            if isthebest:
                result = dataset
            # stop if it is time to finish
            if dataset.nfeatures == 1 or stop:
                break

            # Select features to preserve
            selected_ids = self.__feature_selector(sensitivity)
            # Create a dataset only with selected features
            newdataset = dataset.selectFeatures(selected_ids)

            if not testdataset is None:
                newtestdataset = testdataset.selectFeatures(selected_ids)

            for callable_ in callables:
                callable_(locals())

            # reassign, so in callables we got both older and new
            # datasets
            dataset = newdataset
            if not newtestdataset is None:
                testdataset = dataset

            step += 1
        self["errors"] = errors
        return result

