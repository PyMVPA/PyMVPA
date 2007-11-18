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
                 feature_selector=XPercentFeatureSelector(),
                 stopping_criterion=StopNBackHistoryCriterion(),
                 error_oracle=None
                 ):
        """ Initialize recurse feature elimination
        `sensitivity_analyzer`: `SensitivityAnalyzer`
        `feature_selector`: functor
        `stopping_criterion`: functor
        """

        self.__sensitivity_analyzer = sensitivity_analyzer
        """Sensitivity analyzer used to call at each step"""

        self.__feature_selector = feature_selector
        """Functor which takes care about removing some features"""

        self.__stopping_criterion = stopping_criterion

        self.__error_oracle = error_oracle

        # register some
        self._registerState("errors")


        """
current:
testdata -> independent (split done outside RFE)
dataset -> whole working dataset

need:
test <- outer/final generalization test (never touched till the very end)#
working <- dataset to compute the sensitivity map
itest <- inner test dataset to compute the intermediate generalization error
         determines when to stop RFE

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

    def __call__(self, dataset, testdataset=None, callables=[]):
        """Proceed and select the features recursively eliminating less
        important ones.
        """
        errors = []
        go = True
        result = None
        newtestdataset = None
        step = 0
        while dataset.nfeatures>0:
            # Compute
            sensitivity = self.__sensitivity_analyzer(dataset)
            # Record the error
            errors.append(error_oracle(dataset, testdataset))
            # Check if it is time to stop and if we got
            # the best result
            (go, isthebest) = self.__stopping_criterion(errors)

            if __debug__:
                debug('RFEC',
                      "Step %d: nfeatures=%d error=%.4f best/go=%d/%d" %
                      (step, dataset.nfeatures, errors[-1], isthebest, go))

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

