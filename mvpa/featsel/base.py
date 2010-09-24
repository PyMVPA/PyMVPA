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
from mvpa.base.learner import Learner
from mvpa.base.state import ConditionalAttribute

if __debug__:
    from mvpa.base import debug

class FeatureSelection(Learner):
    """Base class for any feature selection

    Base class for Functors which implement feature selection on the
    datasets.
    """

    selected_ids = ConditionalAttribute(enabled=False)

    def __init__(self, **kwargs):
        # base init first
        Learner.__init__(self, **kwargs)


    def __call__(self, dataset, testdataset=None):
        """Invocation of the feature selection

        Parameters
        ----------
        dataset : Dataset
          dataset used to select features
        testdataset : Dataset
          dataset the might be used to compute a stopping criterion

        Returns
        -------
        Dataset or tuple
          The dataset contains the selected features. If a ``testdataset`` has
          been passed a tuple with both processed datasets is return instead.
          Note that the resulting dataset(s) reference the same values for samples
          attributes (e.g. labels and chunks) of the input dataset(s): be careful
          if you alter them later.
        """
        # Derived classes must provide interface to access other
        # relevant to the feature selection process information (e.g. mask,
        # elimination step (in RFE), etc)
        results = self._call(dataset, testdataset)
        if testdataset is None:
            return results[0]
        else:
            return results



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


    def untrain(self):
        if __debug__:
            debug("FS_", "Untraining sensitivity-based FS: %s" % self)
        self.__sensitivity_analyzer.untrain()


    def _call(self, dataset, testdataset=None):
        """Select the most important features

        Parameters
        ----------
        dataset : Dataset
          used to compute sensitivity maps
        testdataset : Dataset
          optional dataset to select features on

        Returns a tuple of two new datasets with selected feature
        subset of `dataset`.
        """

        sensitivity = self.__sensitivity_analyzer(dataset)
        """Compute the sensitivity map."""

        self.ca.sensitivity = sensitivity

        # Select features to preserve
        selected_ids = self.__feature_selector(sensitivity)

        if __debug__:
            debug("FS_", "Sensitivity: %s Selected ids: %s" %
                  (sensitivity, selected_ids))

        # Create a dataset only with selected features
        wdataset = dataset[:, selected_ids]

        if not testdataset is None:
            wtestdataset = testdataset[:, selected_ids]
        else:
            wtestdataset = None

        # Differ from the order in RFE when actually error reported is for
        results = (wdataset, wtestdataset)

        # WARNING: THIS MUST BE THE LAST THING TO DO ON selected_ids
        selected_ids.sort()
        self.ca.selected_ids = selected_ids

        # dataset with selected features is returned
        return results

    # make it accessible from outside
    sensitivity_analyzer = property(fget=lambda self:self.__sensitivity_analyzer,
                                    doc="Measure which was used to do selection")
