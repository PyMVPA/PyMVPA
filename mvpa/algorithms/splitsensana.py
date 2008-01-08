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

    post = StateVariable(doc="To store results of postprocessing calls")

    def __init__(self, sensana,
                 splitter=NoneSplitter,
                 postproc={},
                 **kwargs):
        """Cheap initialization.

        :Parameters:
            sensana : SensitivityAnalyzer
                that shall be run on the `Dataset` splits.
            splitter : Splitter
                used to split the `Dataset`. By convention the first dataset
                in the tuple returned by the splitter on each iteration is used
                to compute the sensitivity map.
            postproc : dict
                Dictionary of post-processing functors. Each functor will be
                called with the sequence of sensitivity maps. The resulting
                value is then made available via the object's `StateVariable`
                .post, which is a dict, using the respective key from
                `postproc` dictionary.
                Example:
                  postproc={'full': N.array}
                  intermediate = splitana.post['full']
        """
        # init base classes first
        SensitivityAnalyzer.__init__(self, **kwargs)

        self.__sensana = sensana
        """Sensitivity analyzer used to compute the sensitivity maps.
        """
        self.__splitter = splitter
        """Splitter instance used to split the datasets."""
        self.__postproc = postproc
        """Post-processing functors. Each functor will be called with the
        sequence of sensitivity maps. The resulting value is then made
        available via the object's `State` interface using the key stored
        in the `postproc` member.
        """

        self.post = {}
        """Results of postprocessing"""


    def __call__(self, dataset, callables=[]):
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

            # XXX add callbacks to do some magic with the analyzer

            maps.append(sensitivity)

        # do all postprocessing on the sensitivity maps
        for k, v in self.__postproc.iteritems():
            self.post[k] = v(maps)

        # return all maps
        return N.mean(maps, axis=0)

