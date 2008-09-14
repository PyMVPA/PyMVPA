#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Feature selection base class and related stuff base classes and helpers."""

__docformat__ = 'restructuredtext'

from mvpa.featsel.helpers import FractionTailSelector
from mvpa.misc.state import StateVariable, Stateful

if __debug__:
    from mvpa.base import debug

class FeatureSelection(Stateful):
    """Base class for any feature selection

    Base class for Functors which implement feature selection on the
    datasets.
    """

    selected_ids = StateVariable(enabled=False)

    def __init__(self, **kwargs):
        # base init first
        Stateful.__init__(self, **kwargs)


    def __call__(self, dataset, testdataset=None):
        """Invocation of the feature selection

        :Parameters:
          dataset : Dataset
            dataset used to select features
          testdataset : Dataset
            dataset the might be used to compute a stopping criterion

        Returns a tuple with the dataset containing the selected features.
        If present the tuple also contains the selected features of the
        test dataset. Derived classes must provide interface to access other
        relevant to the feature selection process information (e.g. mask,
        elimination step (in RFE), etc)
        """
        raise NotImplementedError



class SensitivityBasedFeatureSelection(FeatureSelection):
    """Feature elimination.

    A `FeaturewiseDatasetMeasure` is used to compute sensitivity maps given a certain
    dataset. These sensitivity maps are in turn used to discard unimportant
    features.
    """

    sensitivity = StateVariable(enabled=False)

    def __init__(self,
                 sensitivity_analyzer,
                 feature_selector=FractionTailSelector(0.05),
                 **kwargs
                 ):
        """Initialize feature selection

        :Parameters:
          sensitivity_analyzer : FeaturewiseDatasetMeasure
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



    def __call__(self, dataset, testdataset=None):
        """Select the most important features

        :Parameters:
          dataset : Dataset
            used to compute sensitivity maps
          testdataset: Dataset
            optional dataset to select features on

        Returns a tuple of two new datasets with selected feature
        subset of `dataset`.
        """

        sensitivity = self.__sensitivity_analyzer(dataset)
        """Compute the sensitivity map."""

        self.sensitivity = sensitivity

        # Select features to preserve
        selected_ids = self.__feature_selector(sensitivity)

        if __debug__:
            debug("FS_", "Sensitivity: %s Selected ids: %s" %
                  (sensitivity, selected_ids))

        # Create a dataset only with selected features
        wdataset = dataset.selectFeatures(selected_ids)

        if not testdataset is None:
            wtestdataset = testdataset.selectFeatures(selected_ids)
        else:
            wtestdataset = None

        # Differ from the order in RFE when actually error reported is for
        results = (wdataset, wtestdataset)

        # WARNING: THIS MUST BE THE LAST THING TO DO ON selected_ids
        selected_ids.sort()
        self.selected_ids = selected_ids

        # dataset with selected features is returned
        return results

    # make it accessible from outside
    sensitivity_analyzer = property(fget=lambda self:self.__sensitivity_analyzer,
                                    doc="Measure which was used to do selection")


class FeatureSelectionPipeline(FeatureSelection):
    """Feature elimination through the list of FeatureSelection's.

    Given as list of FeatureSelections it applies them in turn.
    """

    nfeatures = StateVariable(
        doc="Number of features before each step in pipeline")
    # TODO: may be we should also append resultant number of features?

    def __init__(self,
                 feature_selections,
                 **kwargs
                 ):
        """Initialize feature selection pipeline

        :Parameters:
          feature_selections : lisf of FeatureSelection
            selections which to use. Order matters
        """
        # base init first
        FeatureSelection.__init__(self, **kwargs)

        self.__feature_selections = feature_selections
        """Selectors to use in turn"""


    def __call__(self, dataset, testdataset=None, **kwargs):
        """Invocation of the feature selection
        """
        wdataset = dataset
        wtestdataset = testdataset

        self.selected_ids = None

        self.nfeatures = []
        """Number of features at each step (before running selection)"""

        for fs in self.__feature_selections:

            # enable selected_ids state if it was requested from this class
            fs.states._changeTemporarily(
                enable_states=["selected_ids"], other=self)
            if self.states.isEnabled("nfeatures"):
                self.nfeatures.append(wdataset.nfeatures)

            if __debug__:
                debug('FSPL', 'Invoking %s on (%s, %s)' %
                      (fs, wdataset, wtestdataset))
            wdataset, wtestdataset = fs(wdataset, wtestdataset, **kwargs)

            if self.states.isEnabled("selected_ids"):
                if self.selected_ids == None:
                    self.selected_ids = fs.selected_ids
                else:
                    self.selected_ids = self.selected_ids[fs.selected_ids]

            fs.states._resetEnabledTemporarily()

        return (wdataset, wtestdataset)

    feature_selections = property(fget=lambda self:self.__feature_selections,
                                  doc="List of `FeatureSelections`")
