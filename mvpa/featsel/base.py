# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Feature selection base class and related stuff base classes and helpers."""

__docformat__ = 'restructuredtext'

from mvpa.featsel.helpers import FractionTailSelector, \
                                 NBackHistoryStopCrit, \
                                 BestDetector
from mvpa.mappers.slicing import FeatureSliceMapper
from mvpa.base.state import ConditionalAttribute

if __debug__:
    from mvpa.base import debug


class FeatureSelection(FeatureSliceMapper):
    """Base class for any feature selection

    Base class for Functors which implement feature selection on the
    datasets.
    """
    def __init__(self, **kwargs):
        # initialize FeatureSliceMapper with a `None` slicing argument, since
        # the actual slicing needs to be determined by train()
        FeatureSliceMapper.__init__(self, None, **kwargs)


    def is_mergable(self, other):
        """Returns False

        Unlike simple FeatureSliceMappers feature selection algorithms cannot
        simply be merged. Although it is technical possible to merge actual
        feature selection results, any retraining would yield unexpected
        behavior, since the original training algorithm might have been replaced
        by a Mapper merge.
        """
        return False


    def _untrain(self):
        if __debug__:
            debug("FS_", "Untraining FS: %s" % self)
        self._safe_assign_slicearg(None)
        # ask base class to do its untrain
        super(FeatureSelection, self)._untrain()



class SensitivityBasedFeatureSelection(FeatureSelection):
    """Feature elimination.

    A `FeaturewiseMeasure` is used to compute sensitivity maps given a certain
    dataset. These sensitivity maps are in turn used to discard unimportant
    features.
    """

    sensitivity = ConditionalAttribute(enabled=False)

    def __init__(self,
                 sensitivity_analyzer,
                 feature_selector=FractionTailSelector(0.05),
                 train_analyzer=True,
                 **kwargs
                 ):
        """Initialize feature selection

        Parameters
        ----------
        sensitivity_analyzer : FeaturewiseMeasure
          sensitivity analyzer to come up with sensitivity
        feature_selector : Functor
          Given a sensitivity map it has to return the ids of those
          features that should be kept.
        train_analyzer : bool
          Flag whether to train the sensitivity analyzer on the input dataset
          during train(). If False, the employed sensitivity measure has to be
          already trained before.
        """

        # base init first
        FeatureSelection.__init__(self, **kwargs)

        self.__sensitivity_analyzer = sensitivity_analyzer
        """Sensitivity analyzer to use once"""

        self.__feature_selector = feature_selector
        """Functor which takes care about removing some features."""

        self.__train_analyzer = train_analyzer


    def _train(self, dataset):
        """Select the most important features

        Parameters
        ----------
        dataset : Dataset
          used to compute sensitivity maps
        """
        # optionally train the analyzer first
        if self.__train_analyzer:
            self.__sensitivity_analyzer.train(dataset)

        sensitivity = self.__sensitivity_analyzer(dataset)
        """Compute the sensitivity map."""

        self.ca.sensitivity = sensitivity

        # Select features to preserve
        selected_ids = self.__feature_selector(sensitivity)

        if __debug__:
            debug("FS_", "Sensitivity: %s Selected ids: %s" %
                  (sensitivity, selected_ids))

        # XXX not sure if it really has to be sorted
        selected_ids.sort()
        # announce desired features to the underlying slice mapper
        self._safe_assign_slicearg(selected_ids)
        # and perform its own training
        super(SensitivityBasedFeatureSelection, self)._train(dataset)


    def _untrain(self):
        if __debug__:
            debug("FS_", "Untraining sensitivity-based FS: %s" % self)
        self.__sensitivity_analyzer.untrain()
        # ask base class to do its untrain
        super(SensitivityBasedFeatureSelection, self)._untrain()

    # make it accessible from outside
    sensitivity_analyzer = property(fget=lambda self:self.__sensitivity_analyzer,
                                    doc="Measure which was used to do selection")



class IterativeFeatureSelection(FeatureSelection):
    """
    """
    errors = ConditionalAttribute(
        doc="History of errors")
    nfeatures = ConditionalAttribute(
        doc="History of # of features left")

    def __init__(self,
                 fmeasure,
                 pmeasure,
                 splitter,
                 fselector,
                 stopping_criterion=NBackHistoryStopCrit(BestDetector()),
                 bestdetector=BestDetector(),
                 train_pmeasure=True,
                 **kwargs
                 ):
        """
        Parameters
        ----------
        fmeasure : Measure
          Computed for each candidate feature selection. The measure has
          to compute a scalar value.
        pmeasure : Measure
          Compute against a test dataset for each incremental feature
          set.
        splitter: Splitter
          This splitter instance has to generate at least one dataset split
          when called with the input dataset that is used to compute the
          per-feature criterion for feature selection.
        bestdetector : Functor
          Given a list of error values it has to return a boolean that
          signals whether the latest error value is the total minimum.
        stopping_criterion : Functor
          Given a list of error values it has to return whether the
          criterion is fulfilled.
        fselector : Functor
        train_clf : bool
          Flag whether the classifier in `transfer_error` should be
          trained before computing the error. In general this is
          required, but if the `sensitivity_analyzer` and
          `transfer_error` share and make use of the same classifier it
          can be switched off to save CPU cycles. Default `None` checks
          if sensitivity_analyzer is based on a classifier and doesn't train
          if so.
        """
        # bases init first
        FeatureSelection.__init__(self, **kwargs)

        self._fmeasure = fmeasure
        self._pmeasure = pmeasure
        self._splitter = splitter
        self._fselector = fselector
        self._stopping_criterion = stopping_criterion
        self._bestdetector = bestdetector
        self._train_pmeasure = train_pmeasure


    def _untrain(self):
        if __debug__:
            debug("FS_", "Untraining Iterative FS: %s" % self)
        self._fmeasure.untrain()
        self._pmeasure.untrain()
        # ask base class to do its untrain
        super(IterativeFeatureSelection, self)._untrain()


    def _evaluate_pmeasure(self, train, test):
        # local binding
        pmeasure = self._pmeasure
        # might safe some cycles to prevent training the measure, but only
        # the user can know whether this is sensible or possible
        if self._train_pmeasure:
            pmeasure.train(train)
        # actually run the performance measure to estimate "quality" of
        # selection
        return pmeasure(test)


    def _get_traintest_ds(self, ds):
        # activate the dataset splitter
        dsgen = self._splitter.generate(ds)
        # and derived the dataset part that is used for computing the selection
        # criterion
        trainds = dsgen.next()
        testds = dsgen.next()
        return trainds, testds


