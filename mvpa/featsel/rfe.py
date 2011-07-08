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

from mvpa.support.copy import copy
from mvpa.clfs.transerror import ClassifierError
from mvpa.measures.base import Sensitivity
from mvpa.featsel.base import IterativeFeatureSelection
from mvpa.featsel.helpers import BestDetector, \
                                 NBackHistoryStopCrit, \
                                 FractionTailSelector
import numpy as np
from mvpa.base.state import ConditionalAttribute

if __debug__:
    from mvpa.base import debug

# TODO: Abs value of sensitivity should be able to rule RFE
# Often it is what abs value of the sensitivity is what matters.
# So we should either provide a simple decorator around arbitrary
# FeatureSelector to convert sensitivities to abs values before calling
# actual selector, or a decorator around SensitivityEstimators


class RFE(IterativeFeatureSelection):
    """Recursive feature elimination.

    A `FeaturewiseMeasure` is used to compute sensitivity maps given a
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

    Examples
    --------

    There are multiple possible ways to design an RFE.  Here is one
    example which would rely on a SplitClassifier to extract
    sensitivities and provide estimate of performance (error)

    >>> # Lazy import
    >>> from mvpa.suite import *
    >>> rfesvm_split = SplitClassifier(LinearCSVMC(), OddEvenPartitioner())
    >>> # design an RFE feature selection to be used with a classifier
    >>> rfe = RFE(rfesvm_split.get_sensitivity_analyzer(
    ...              # take sensitivities per each split, L2 norm, mean, abs them
    ...              postproc=ChainMapper([ FxMapper('features', l2_normed),
    ...                                     FxMapper('samples', np.mean),
    ...                                     FxMapper('samples', np.abs)])),
    ...           # use the error stored in the confusion matrix of split classifier
    ...           ConfusionBasedError(rfesvm_split, confusion_state='stats'),
    ...           # we just extract error from confusion, so need to split dataset
    ...           Repeater(2),
    ...           # select 50% of the best on each step
    ...           fselector=FractionTailSelector(
    ...               0.50,
    ...               mode='select', tail='upper'),
    ...           # and stop whenever error didn't improve for up to 10 steps
    ...           stopping_criterion=NBackHistoryStopCrit(BestDetector(), 10),
    ...           # we just extract it from existing confusion
    ...           train_pmeasure=False,
    ...           # but we do want to update sensitivities on each step
    ...           update_sensitivity=True)
    >>> clf = \
    ...  FeatureSelectionClassifier(
    ...   LinearCSVMC(),
    ...   # on features selected via RFE
    ...   rfe,
    ...   # custom description
    ...   descr='LinSVM+RFE(splits_avg)' )

    """

    history = ConditionalAttribute(
        doc="Last step # when each feature was still present")
    sensitivities = ConditionalAttribute(enabled=False,
        doc="History of sensitivities (might consume too much memory")

    def __init__(self,
                 fmeasure,
                 pmeasure,
                 splitter,
                 fselector=FractionTailSelector(0.05),
                 update_sensitivity=True,
                 **kwargs):
        # XXX Allow for multiple stopping criterions, e.g. error not decreasing
        # anymore OR number of features less than threshold
        """Initialize recursive feature elimination

        Parameters
        ----------
        fmeasure : FeaturewiseMeasure
        pmeasure : Measure
          used to compute the transfer error of a classifier based on a
          certain feature set on the test dataset.
          NOTE: If sensitivity analyzer is based on the same
          classifier as transfer_error is using, make sure you
          initialize transfer_error with train=False, otherwise
          it would train classifier twice without any necessity.
        splitter: Splitter
          This splitter instance has to generate at least two dataset splits
          when called with the input dataset. The first split serves as the
          training dataset and the second as the evaluation dataset.
        fselector : Functor
          Given a sensitivity map it has to return the ids of those
          features that should be kept.
        update_sensitivity : bool
          If False the sensitivity map is only computed once and reused
          for each iteration. Otherwise the senstitivities are
          recomputed at each selection step.
        """
        # bases init first
        IterativeFeatureSelection.__init__(self, fmeasure, pmeasure, splitter,
                                           fselector, **kwargs)

        self.__update_sensitivity = update_sensitivity
        """Flag whether sensitivity map is recomputed for each step."""


    def _train(self, ds):
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
        # get the initial split into train and test
        dataset, testdataset = self._get_traintest_ds(ds)

        if __debug__:
            debug('RFEC',
                  "Initiating RFE with training on %s and testing using %s",
                  (dataset, testdataset))
        errors = []
        """Computed error for each tested features set."""

        ca = self.ca
        ca.nfeatures = []
        """Number of features at each step. Since it is not used by the
        algorithm it is stored directly in the conditional attribute"""

        ca.history = np.arange(dataset.nfeatures)
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

        orig_feature_ids = np.arange(dataset.nfeatures)
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
                sensitivity = self._fmeasure(wdataset)
                if len(sensitivity) > 1:
                    raise ValueError(
                            "RFE cannot handle multiple sensitivities at once. "
                            "'%s' returned %i sensitivities."
                            % (self._fmeasure.__class__.__name__,
                               len(sensitivity)))

            if ca.is_enabled("sensitivities"):
                ca.sensitivities.append(sensitivity)

            # get error for current feature set (handles optional retraining)
            error = self._evaluate_pmeasure(wdataset, wtestdataset)
            # Record the error
            errors.append(np.asscalar(error))

            # Check if it is time to stop and if we got
            # the best result
            stop = self._stopping_criterion(errors)
            isthebest = self._bestdetector(errors)

            nfeatures = wdataset.nfeatures

            if ca.is_enabled("nfeatures"):
                ca.nfeatures.append(wdataset.nfeatures)

            # store result
            if isthebest:
                result_selected_ids = orig_feature_ids

            if __debug__:
                debug('RFEC',
                      "Step %d: nfeatures=%d error=%.4f best/stop=%d/%d " %
                      (step, nfeatures, np.asscalar(error), isthebest, stop))

            # stop if it is time to finish
            if nfeatures == 1 or stop:
                break

            # Select features to preserve
            selected_ids = self._fselector(sensitivity)

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

            # we already have the initial sensitivities, so even for a shared
            # classifier we can cleanup here
            self._pmeasure.untrain()
        # charge conditional attributes
        self.ca.errors = errors
        self.ca.selected_ids = result_selected_ids
        if __debug__:
            debug('RFEC',
                  "Selected %d features: %s",
                  (len(result_selected_ids), result_selected_ids))

        # announce desired features to the underlying slice mapper
        # do copy to survive later selections
        self._safe_assign_slicearg(copy(result_selected_ids))
