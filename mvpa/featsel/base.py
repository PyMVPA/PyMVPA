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

from mvpa.featsel.helpers import FractionTailSelector
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

        """

        # base init first
        FeatureSelection.__init__(self, **kwargs)

        self.__sensitivity_analyzer = sensitivity_analyzer
        """Sensitivity analyzer to use once"""

        self.__feature_selector = feature_selector
        """Functor which takes care about removing some features."""


    def _train(self, dataset):
        """Select the most important features

        Parameters
        ----------
        dataset : Dataset
          used to compute sensitivity maps
        """
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


    def untrain(self):
        if __debug__:
            debug("FS_", "Untraining sensitivity-based FS: %s" % self)
        self.__sensitivity_analyzer.untrain()
        # reset slicearg that has been assigned during training
        self._safe_assign_slicearg(None)
        # ask base class to do its untrain
        super(SensitivityBasedFeatureSelection, self).untrain()

    # make it accessible from outside
    sensitivity_analyzer = property(fget=lambda self:self.__sensitivity_analyzer,
                                    doc="Measure which was used to do selection")
