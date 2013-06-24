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

import numpy as np

from mvpa.featsel.helpers import FractionTailSelector
from mvpa.misc.state import StateVariable, ClassWithCollections

if __debug__:
    from mvpa.base import debug

class FeatureSelection(ClassWithCollections):
    """Base class for any feature selection

    Base class for Functors which implement feature selection on the
    datasets.
    """

    selected_ids = StateVariable(enabled=False)

    def __init__(self, **kwargs):
        # base init first
        ClassWithCollections.__init__(self, **kwargs)


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


    def untrain(self):
        """ 'Untrain' feature selection

        Necessary for full 'untraining' of the classifiers. By default
        does nothing, needs to be overridden in corresponding feature
        selections to pass to the sensitivities
        """
        pass


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


    def untrain(self):
        if __debug__:
            debug("FS_", "Untraining sensitivity-based FS: %s" % self)
        self.__sensitivity_analyzer.untrain()


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
        if not selected_ids.flags.writeable:
            # With numpy 1.7 sometimes it returns R/O arrays... not clear yet why.
            # Dirty fix: work on a copy
            selected_ids = np.sort(selected_ids)
        else:
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


    def untrain(self):
        if __debug__:
            debug("FS_", "Untraining FS pipeline: %s" % self)
        for fs in self.__feature_selections:
            fs.untrain()


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



class CombinedFeatureSelection(FeatureSelection):
    """Meta feature selection utilizing several embedded selection methods.

    Each embedded feature selection method is computed individually. Afterwards
    all feature sets are combined by either taking the union or intersection of
    all sets.

    The individual feature sets of all embedded methods are optionally avialable
    from the `selections_ids` state variable.
    """
    selections_ids = StateVariable(
        doc="List of feature id sets for each performed method.")

    def __init__(self, feature_selections, combiner, **kwargs):
        """
        :Parameters:
          feature_selections: list
            FeatureSelection instances to run. Order is not important.
          combiner: 'union', 'intersection'
            which method to be used to combine the feature selection set of
            all computed methods.
        """
        FeatureSelection.__init__(self, **kwargs)

        self.__feature_selections = feature_selections
        self.__combiner = combiner


    def untrain(self):
        if __debug__:
            debug("FS_", "Untraining combined FS: %s" % self)
        for fs in self.__feature_selections:
            fs.untrain()


    def __call__(self, dataset, testdataset=None):
        """Really run it.
        """
        # to hold the union
        selected_ids = None
        # to hold the individuals
        self.selections_ids = []

        for fs in self.__feature_selections:
            # we need the feature ids that were selection by each method,
            # so enable them temporarily
            fs.states._changeTemporarily(
                enable_states=["selected_ids"], other=self)

            # compute feature selection, but ignore return datasets
            fs(dataset, testdataset)

            # retrieve feature ids and determined union of all selections
            if selected_ids == None:
                selected_ids = set(fs.selected_ids)
            else:
                if self.__combiner == 'union':
                    selected_ids.update(fs.selected_ids)
                elif self.__combiner == 'intersection':
                    selected_ids.intersection_update(fs.selected_ids)
                else:
                    raise ValueError, "Unknown combiner '%s'" % self.__combiner

            # store individual set in state
            self.selections_ids.append(fs.selected_ids)

            # restore states to previous settings
            fs.states._resetEnabledTemporarily()

        # finally apply feature set union selection to original datasets
        selected_ids = sorted(list(selected_ids))

        # take care of optional second dataset
        td_sel = None
        if not testdataset is None:
            td_sel = testdataset.selectFeatures(self.selected_ids)

        # and main dataset
        d_sel = dataset.selectFeatures(selected_ids)

        # finally store ids in state
        self.selected_ids = selected_ids

        return (d_sel, td_sel)


    feature_selections = property(fget=lambda self:self.__feature_selections,
                                  doc="List of `FeatureSelections`")
    combiner = property(fget=lambda self:self.__combiner,
                        doc="Selection set combination method.")
