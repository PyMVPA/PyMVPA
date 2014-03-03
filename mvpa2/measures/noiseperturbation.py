# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Derive sensitivity maps for a metric by selective noise perturbation"""

__docformat__ = 'restructuredtext'

if __debug__:
    from mvpa2.base import debug

from mvpa2.support.copy import deepcopy

import numpy as np

from mvpa2.measures.base import FeaturewiseMeasure
from mvpa2.datasets.base import Dataset


class NoisePerturbationSensitivity(FeaturewiseMeasure):
    """Sensitivity based on the effect of noise perturbation on a measure.

    This is a `FeaturewiseMeasure` that uses a scalar `Measure`
    and selective noise perturbation to compute a sensitivity map.

    First the scalar `Measure` computed using the original dataset. Next
    the data measure is computed multiple times each with a single feature in
    the dataset perturbed by noise. The resulting difference in the
    scalar `Measure` is used as the sensitivity for the respective
    perturbed feature. Large differences are treated as an indicator of a
    feature having great impact on the scalar `Measure`.

    Notes
    -----
    The computed sensitivity map might have positive and negative values!
    """
    is_trained = True
    """Indicate that this measure is always trained."""

    def __init__(self, datameasure,
                 noise=np.random.normal):
        """
        Parameters
        ----------
        datameasure : `Measure`
          Used to quantify the effect of noise perturbation.
        noise: Callable
          Used to generate noise. The noise generator has to return an 1d array
          of n values when called the `size=n` keyword argument. This is the
          default interface of the random number generators in NumPy's
          `random` module.
        """
        # init base classes first
        FeaturewiseMeasure.__init__(self)

        self.__datameasure = datameasure
        self.__noise = noise


    def _call(self, dataset):
        # first cast to floating point dtype, because noise is most likely
        # floating point as well and '+=' on int would not do the right thing
        if not np.issubdtype(dataset.samples.dtype, np.float):
            ds = dataset.copy(deep=False)
            ds.samples = dataset.samples.astype('float32')
            dataset = ds

        if __debug__:
            nfeatures = dataset.nfeatures

        # using a list here, to be able to handle output of unknown
        # dimensionality
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

            # store current feature to restore it later on
            current_feature = dataset.samples[:, feature].copy()

            # add noise to current feature
            dataset.samples[:, feature] += self.__noise(size=len(dataset))

            # compute the datameasure on the perturbed dataset
            perturbed_measure = self.__datameasure(dataset)

            # restore the current feature
            dataset.samples[:, feature] = current_feature

            # difference from original datameasure is sensitivity
            sens_map.append(perturbed_measure.samples - orig_measure.samples)

        if __debug__:
            debug('PSA', '')

        # turn into an array and get rid of unnecessary axes -- ideally yielding
        # 2D array
        sens_map = np.array(sens_map).squeeze()
        # swap first to axis: we have nfeatures on first but want it as second
        # in a dataset
        sens_map = np.swapaxes(sens_map, 0, 1)
        return Dataset(sens_map)
