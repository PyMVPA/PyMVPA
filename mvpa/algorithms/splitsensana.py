#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""This is a `SensitivityAnalyzer` that uses another `SensitivityAnalyzer`
and runs it multiple times on differents splits of a `Dataset`.
"""

__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.algorithms.datameasure import SensitivityAnalyzer
from mvpa.datasets.splitter import NoneSplitter
from mvpa.misc.state import StateVariable
from mvpa.misc.transformers import FirstAxisMean

class SplittingSensitivityAnalyzer(SensitivityAnalyzer):
    """This is a `SensitivityAnalyzer` that uses another `SensitivityAnalyzer`
    and runs it multiple times on differents splits of a `Dataset`.

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
            sensana : SensitivityAnalyzer
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
        SensitivityAnalyzer.__init__(self, **kwargs)

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

        self.maps = maps
        """Store the maps across splits"""

        # return all maps
        return self.__combiner(maps)
