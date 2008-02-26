#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""This is a `SensitivityAnalyzer` that uses a `ScalarDatasetMeasure` and
selective noise perturbation to compute a sensitivity map.
"""

__docformat__ = 'restructuredtext'

if __debug__:
    from mvpa.misc import debug

from copy import deepcopy

import numpy as N

from mvpa.algorithms.datameasure import SensitivityAnalyzer


class PerturbationSensitivityAnalyzer(SensitivityAnalyzer):
    """This is a `SensitivityAnalyzer` that uses a `ScalarDatasetMeasure` and
    selective noise perturbation to compute a sensitivity map.

    First the `ScalarDatasetMeasure` computed using the original dataset. Next
    the data measure is computed multiple times each with a single feature in
    the dataset perturbed by noise. The resulting difference in the
    `ScalarDatasetMeasure` is used as the sensitivity for the respective
    perturbed feature. Large differences are treated as an indicator of a
    feature having great impact on the `ScalarDatasetMeasure`.

    The computed sensitivity map might have positive and negative values!
    """
    def __init__(self, datameasure,
                 noise=N.random.normal):
        """Cheap initialization.

        Parameters
        ----------
        - `datameasure`: `Datameasure` that is used to quantify the effect of
                         noise perturbation.
        - `noise`: Functor to generate noise. The noise generator has to return
                   an 1d array of n values when called the `size=n` keyword
                   argument. This is the default interface of the random number
                   generators in NumPy's `random` module.
        """
        # init base classes first
        SensitivityAnalyzer.__init__(self)

        self.__datameasure = datameasure
        self.__noise = noise


    def _call(self, dataset):
        """Compute the sensitivity map.

        Returns a 1d array of sensitivities for all features in `dataset`.
        """
        # first cast to floating point dtype, because noise is most likely
        # floating point as well and '+=' on int would not do the right thing
        # XXX should we already deepcopy here to keep orig dtype?
        if not N.issubdtype(dataset.samples.dtype, N.float):
            dataset.setSamplesDType('float32')

        if __debug__:
            nfeatures = dataset.nfeatures

        sens_map = []

        # compute the datameasure on the original dataset
        # this is used as a baseline
        orig_measure = self.__datameasure(dataset)

        # do for every _single_ feature in the dataset
        for feature in xrange(dataset.nfeatures):
            if __debug__:
                debug('PSA', "Analyzing %i features: %i [%i%%]" \
                    % (nfeatures,
                       feature+1,
                       float(feature+1)/nfeatures*100,), cr=True)

            # make a copy of the dataset to preserve data integrity
            wdata = deepcopy(dataset)

            # add noise to current feature
            wdata.samples[:, feature] += self.__noise(size=wdata.nsamples)

            # compute the datameasure on the perturbed dataset
            perturbed_measure = self.__datameasure(wdata)

            # difference from original datameasure is sensitivity
            sens_map.append(perturbed_measure - orig_measure)

        if __debug__:
            debug('PSA', '')

        return N.array(sens_map)

