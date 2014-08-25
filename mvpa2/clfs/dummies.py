# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Collection of dummy (e.g. random) classifiers.  Primarily for testing.
"""

__docformat__ = 'restructuredtext'

import numpy as np
import numpy.random as npr

from mvpa2.base.param import Parameter
from mvpa2.base.types import accepts_dataset_as_samples, is_datasetlike
from mvpa2.clfs.base import Classifier

__all__ = ['Classifier', 'SameSignClassifier', 'RandomClassifier',
           'Less1Classifier']

#
# Few silly classifiers
#
class SameSignClassifier(Classifier):
    """Dummy classifier which reports +1 class if both features have
    the same sign, -1 otherwise"""

    __tags__ = ['notrain2predict']
    def __init__(self, **kwargs):
        Classifier.__init__(self, **kwargs)

    def _train(self, data):
        # we don't need that ;-)
        pass

    @accepts_dataset_as_samples
    def _predict(self, data):
        data = np.asanyarray(data)
        datalen = len(data)
        estimates = []
        for d in data:
            estimates.append(2*int( (d[0]>=0) == (d[1]>=0) )-1)
        self.ca.predictions = estimates
        self.ca.estimates = estimates            # just for the sake of having estimates
        return estimates


class RandomClassifier(Classifier):
    """Dummy classifier deciding on labels absolutely randomly
    """

    __tags__ = ['random', 'non-deterministic']

    same = Parameter(
        False, constraints='bool',
        doc="If a dataset arrives to predict, assign identical (but random) label "
            "to all samples having the same label in original, thus mimiquing the "
            "situation where testing samples are not independent.")

    def __init__(self, **kwargs):
        Classifier.__init__(self, **kwargs)
        self._ulabels = None

    def _train(self, data):
        self._ulabels = data.sa[self.get_space()].unique

    @accepts_dataset_as_samples
    def _predict(self, data):
        l = len(self._ulabels)
        # oh those lovely random estimates, for now just an estimate
        # per sample. Since we are random after all -- keep it random
        self.ca.estimates = np.random.normal(size=len(data))
        if is_datasetlike(data) and self.params.same:
            # decide on mapping between original labels
            labels_map = dict(
                (t, rt) for t, rt in zip(self._ulabels,
                                         self._ulabels[npr.randint(0, l, size=l)]))
            return [labels_map[t] for t in data.sa[self.get_space()].value]
        else:
            # random one per each
            return self._ulabels[npr.randint(0, l, size=len(data))]


class Less1Classifier(SameSignClassifier):
    """Dummy classifier which reports +1 class if abs value of max less than 1"""
    def _predict(self, data):
        datalen = len(data)
        estimates = []
        for d in data:
            estimates.append(2*int(max(d)<=1)-1)
        self.predictions = estimates
        return estimates
