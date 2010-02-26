# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Recursive feature elimination."""

__docformat__ = 'restructuredtext'

from mvpa.clfs.transerror import ClassifierError
from mvpa.measures.base import Sensitivity
from mvpa.featsel.base import FeatureSelection
from mvpa.featsel.helpers import BestDetector, \
                                 NBackHistoryStopCrit, \
                                 FractionTailSelector
from numpy import arange
from mvpa.misc.state import StateVariable

if __debug__:
    from mvpa.base import debug

# TODO: Abs value of sensitivity should be able to rule RFE
# Often it is what abs value of the sensitivity is what matters.
# So we should either provide a simple decorator around arbitrary
# FeatureSelector to convert sensitivities to abs values before calling
# actual selector, or a decorator around SensitivityEstimators


class RFE(FeatureSelection):
    """Recursive feature elimination.

    A `FeaturewiseDatasetMeasure` is used to compute sensitivity maps given a
    certain dataset. These sensitivity maps are in turn used to discard
    unimportant features. For each feature selection the transfer error on some
    testdatset is computed. This procedure is repeated until a given
    `StoppingCriterion` is reached.

    References
    ----------
    Such strategy after
      Guyon, I., Weston, J., Barnhill, S., & Vapnik, V. (2002). Gene
      selection for cancer classification using support vector
      machines. Mach. Learn., 46(1-3), 389--422.
    was applied to SVM-based analysis of fMRI data in
      Hanson, S. J. & Halchenko, Y. O. (2008). Brain reading using
      full brain support vector machines for object recognition:
      there is no "face identification area". Neural Computation, 20,
      486--503.
    """

    errors = StateVariable(
        doc="History of errors through RFE")
    nfeatures = StateVariable(
        doc="History of # of features left")
    history = StateVariable(
        doc="Last step # when each feature was still present")
    sensitivities = StateVariable(enabled=False,
        doc="History of sensitivities (might consume too much memory")

    def __init__(self,
                 sensitivity_analyzer,
                 transfer_error,
                 feature_selector=FractionTailSelector(0.05),
                 bestdetector=BestDetector(),
                 stopping_criterion=NBackHistoryStopCrit(BestDetector()),
                 train_clf=None,
                 update_sensitivity=True,
                 **kargs
                 ):
        # XXX Allow for multiple stopping criterions, e.g. error not decreasing
        # anymore OR number of features less than threshold
        """Initialize recursive feature elimination

        Parameters
        ----------
        sensitivity_analyzer : FeaturewiseDatasetMeasure object
        transfer_error : TransferError object
          used to compute the transfer error of a classifier based on a
          certain feature set on the test dataset.
          NOTE: If sensitivity analyzer is based on the same
          classifier as transfer_error is using, make sure you
          initialize transfer_error with train=False, otherwise
          it would train classifier twice without any necessity.
        feature_selector : Functor
          Given a sensitivity map it has to return the ids of those
          features that should be kept.
        bestdetector : Functor
          Given a list of error values it has to return a boolean that
          signals whether the latest error value is the total minimum.
        stopping_criterion : Functor
          Given a list of error values it has to return whether the
          criterion is fulfilled.
        train_clf : bool
          Flag whether the classifier in `transfer_error` should be
          trained before computing the error. In general this is
          required, but if the `sensitivity_analyzer` and
          `transfer_error` share and make use of the same classifier it
          can be switched off to save CPU cycles. Default `None` checks
          if sensitivity_analyzer is based on a classifier and doesn't train
          if so.
        update_sensitivity : bool
          If False the sensitivity map is only computed once and reused
          for each iteration. Otherwise the senstitivities are
          recomputed at each selection step.
        """

        # base init first
        FeatureSelection.__init__(self, **kargs)

        self.__sensitivity_analyzer = sensitivity_analyzer
        """Sensitivity analyzer used to call at each step."""

        self.__transfer_error = transfer_error
        """Compute transfer error for each feature set."""

        self.__feature_selector = feature_selector
        """Functor which takes care about removing some features."""

        self.__stopping_criterion = stopping_criterion

        self.__bestdetector = bestdetector

        if train_clf is None:
            self.__train_clf = isinstance(sensitivity_analyzer,
                                          Sensitivity)
        else:
            self.__train_clf = train_clf
            """Flag whether training classifier is required."""

        self.__update_sensitivity = update_sensitivity
        """Flag whether sensitivity map is recomputed for each step."""

        # force clf training when sensitivities are not updated as otherwise
        # shared classifiers are not retrained
        if not self.__update_sensitivity \
               and isinstance(self.__transfer_error, ClassifierError) \
               and not self.__train_clf:
            if __debug__:
                debug("RFEC", "Forcing training of classifier since " +
                      "sensitivities aren't updated at each step")
            self.__train_clf = True


    def _call(self, dataset, testdataset):
        """Proceed and select the features recursively eliminating less
        important ones.

        Parameters
        ----------
        dataset : Dataset
          used to compute sensitivity maps and train a classifier
          to determine the transfer error
        testdataset : Dataset
          used to test the trained classifer to determine the
          transfer error

        Returns a tuple of two new datasets with the feature subset of
        `dataset` that had the lowest transfer error of all tested
        sets until the stopping criterion was reached. The first
        dataset is the feature subset of the training data and the
        second the selection of the test dataset.
        """
        errors = []
        """Computed error for each tested features set."""

        ca = self.ca
        ca.nfeatures = []
        """Number of features at each step. Since it is not used by the
        algorithm it is stored directly in the state variable"""

        ca.history = arange(dataset.nfeatures)
        """Store the last step # when the feature was still present
        """

        ca.sensitivities = []

        stop = False
        """Flag when RFE should be stopped."""

        results = None
        """Will hold the best feature set ever."""

        wdataset = dataset
        """Operate on working dataset initially identical."""

        wtestdataset = testdataset
        """Same feature selection has to be performs on test dataset as well.
        This will hold the current testdataset."""

        step = 0
        """Counter how many selection step where done."""

        orig_feature_ids = arange(dataset.nfeatures)
        """List of feature Ids as per original dataset remaining at any given
        step"""

        sensitivity = None
        """Contains the latest sensitivity map."""

        result_selected_ids = orig_feature_ids
        """Resultant ids of selected features. Since the best is not
        necessarily is the last - we better keep this one around. By
        default -- all features are there"""
        selected_ids = result_selected_ids

        while wdataset.nfeatures > 0:

            if __debug__:
                debug('RFEC',
                      "Step %d: nfeatures=%d" % (step, wdataset.nfeatures))

            # mark the features which are present at this step
            # if it brings anyb mentionable computational burden in the future,
            # only mark on removed features at each step
            ca.history[orig_feature_ids] = step

            # Compute sensitivity map
            if self.__update_sensitivity or sensitivity == None:
                sensitivity = self.__sensitivity_analyzer(wdataset)
                if len(sensitivity) > 1:
                    raise ValueError(
                            "RFE cannot handle multiple sensitivities at once. "
                            "'%s' returned %i sensitivities."
                            % (self.__sensitivity_analyzer.__class__.__name__,
                               len(sensitivity)))

            if ca.is_enabled("sensitivities"):
                ca.sensitivities.append(sensitivity)

            # do not retrain clf if not necessary
            if self.__train_clf:
                error = self.__transfer_error(wtestdataset, wdataset)
            else:
                error = self.__transfer_error(wtestdataset, None)

            # Record the error
            errors.append(error)

            # Check if it is time to stop and if we got
            # the best result
            stop = self.__stopping_criterion(errors)
            isthebest = self.__bestdetector(errors)

            nfeatures = wdataset.nfeatures

            if ca.is_enabled("nfeatures"):
                ca.nfeatures.append(wdataset.nfeatures)

            # store result
            if isthebest:
                results = (wdataset, wtestdataset)
                result_selected_ids = orig_feature_ids

            if __debug__:
                debug('RFEC',
                      "Step %d: nfeatures=%d error=%.4f best/stop=%d/%d " %
                      (step, nfeatures, error, isthebest, stop))

            # stop if it is time to finish
            if nfeatures == 1 or stop:
                break

            # Select features to preserve
            selected_ids = self.__feature_selector(sensitivity)

            if __debug__:
                debug('RFEC_',
                      "Sensitivity: %s, nfeatures_selected=%d, selected_ids: %s" %
                      (sensitivity, len(selected_ids), selected_ids))


            # Create a dataset only with selected features
            wdataset = wdataset[:, selected_ids]

            # select corresponding sensitivity values if they are not
            # recomputed
            if not self.__update_sensitivity:
                sensitivity = sensitivity[selected_ids]

            # need to update the test dataset as well
            # XXX why should it ever become None?
            # yoh: because we can have __transfer_error computed
            #      using wdataset. See xia-generalization estimate
            #      in lightsvm. Or for god's sake leave-one-out
            #      on a wdataset
            # TODO: document these cases in this class
            if not testdataset is None:
                wtestdataset = wtestdataset[:, selected_ids]

            step += 1

            # WARNING: THIS MUST BE THE LAST THING TO DO ON selected_ids
            selected_ids.sort()
            if self.ca.is_enabled("history") \
                   or self.ca.is_enabled('selected_ids'):
                orig_feature_ids = orig_feature_ids[selected_ids]


            if hasattr(self.__transfer_error, "clf"):
                self.__transfer_error.clf.untrain()
        # charge state variables
        self.ca.errors = errors
        self.ca.selected_ids = result_selected_ids

        # best dataset ever is returned
        return results

