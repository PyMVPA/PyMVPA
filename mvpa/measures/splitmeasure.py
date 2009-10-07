# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""This is a `FeaturewiseDatasetMeasure` that uses another
`FeaturewiseDatasetMeasure` and runs it multiple times on differents splits of
a `Dataset`.
"""

__docformat__ = 'restructuredtext'

import numpy as N
from mvpa.measures.base import FeaturewiseDatasetMeasure
from mvpa.datasets.splitters import NoneSplitter
from mvpa.misc.state import StateVariable
from mvpa.misc.transformers import FirstAxisMean

if __debug__:
    from mvpa.base import debug


class SplitFeaturewiseMeasure(FeaturewiseDatasetMeasure):
    """This is a `FeaturewiseDatasetMeasure` that uses another
    `FeaturewiseDatasetMeasure` and runs it multiple times on differents
    splits of a `Dataset`.

    When called with a `Dataset` it returns the mean sensitivity maps of all
    data splits.

    Additonally this class supports the `State` interface. Several
    postprocessing functions can be specififed to the constructor. The results
    of the functions specified in the `postproc` dictionary will be available
    via their respective keywords.
    """

    maps = StateVariable(enabled=False,
                         doc="To store maps per each split")

    def __init__(self, sensana,
                 splitter=NoneSplitter,
                 combiner=FirstAxisMean,
                 **kwargs):
        """Cheap initialization.

        :Parameters:
            sensana : FeaturewiseDatasetMeasure
                that shall be run on the `Dataset` splits.
            splitter : Splitter
                used to split the `Dataset`. By convention the first dataset
                in the tuple returned by the splitter on each iteration is used
                to compute the sensitivity map.
            combiner
                This functor will be called on an array of sensitivity maps
                and the result will be returned by __call__(). The result of
                a combiner must be an 1d ndarray.
        """
        # init base classes first
        FeaturewiseDatasetMeasure.__init__(self, **kwargs)

        self.__sensana = sensana
        """Sensitivity analyzer used to compute the sensitivity maps.
        """
        self.__splitter = splitter
        """Splitter instance used to split the datasets."""
        self.__combiner = combiner
        """Function to combine sensitivities to serve a result of
        __call__()"""


    def _call(self, dataset):
        """Compute sensitivity maps for all dataset splits and run the
        postprocessing functions afterward (if any).

        Returns a list of all computed sensitivity maps. Postprocessing results
        are available via the objects `State` interface.
        """

        maps = []

        # splitter
        for split in self.__splitter(dataset):
            # compute sensitivity using first dataset in split
            sensitivity = self.__sensana(split[0])

            maps.append(sensitivity)

        self.states.maps = maps
        """Store the maps across splits"""

        # return all maps
        return self.__combiner(maps)



class TScoredFeaturewiseMeasure(SplitFeaturewiseMeasure):
    """`SplitFeaturewiseMeasure` computing featurewise t-score of
    sensitivities across splits.
    """
    def __init__(self, sensana, splitter, noise_level=0.0, **kwargs):
        """Cheap initialization.

        :Parameters:
            sensana : SensitivityAnalyzer
                that shall be run on the `Dataset` splits.
            splitter : Splitter
                used to split the `Dataset`. By convention the first dataset
                in the tuple returned by the splitter on each iteration is used
                to compute the sensitivity map.
            noise_level: float
                Theoretical output of the respective `SensitivityAnalyzer`
                for a pure noise pattern. For most algorithms this is probably
                zero, hence the default.
        """
        # init base classes first
        #  - get full sensitivity maps from SplittingSensitivityAnalyzer
        #  - no postprocessing
        #  - leave States handling to base class
        SplitFeaturewiseMeasure.__init__(self,
                                              sensana,
                                              splitter,
                                              combiner=N.array,
                                              **kwargs)

        self.__noise_level = noise_level
        """Output of the sensitivity analyzer when there is no signal."""


    def _call(self, dataset, callables=[]):
        """Compute sensitivity maps for all dataset splits and return the
        featurewise t-score of them.
        """
        # let base class compute the sensitivity maps
        maps = SplitFeaturewiseMeasure._call(self, dataset)

        # feature wise mean
        m = N.mean(maps, axis=0)
        #m = N.min(maps, axis=0)
        # featurewise variance
        v = N.var(maps, axis=0)
        # degrees of freedom (n-1 for one-sample t-test)
        df = maps.shape[0] - 1

        # compute t-score
        t = (m - self.__noise_level) / N.sqrt(v * (1.0 / maps.shape[0]))

        if __debug__:
            debug('SA', 'T-score sensitivities computed for %d maps ' %
                  maps.shape[0] +
                  'min=%f max=%f. mean(m)=%f mean(v)=%f  Result min=%f max=%f mean(abs)=%f' %
                  (N.min(maps), N.max(maps), N.mean(m), N.mean(v), N.min(t),
                   N.max(t), N.mean(N.abs(t))))

        return t
