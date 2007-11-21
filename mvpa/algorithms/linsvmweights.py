#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""""""

__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.algorithms.datameasure import SensitivityAnalyzer
from mvpa.clf.svm import LinearSVM


class LinearSVMWeights(SensitivityAnalyzer):
    """`SensitivityAnalyzer` that reports the weights of a linear SVM trained
    on a given `Dataset`.
    """
    def __init__(self, clf):
        """Initialize the analyzer with the classifier it shall use.

        Parameters
        ----------
        - `clf`: Classifier instance. Only classifiers sub-classed from
                 `LinearSVM` may be used.
        """
        if not isinstance(clf, LinearSVM):
            raise ValueError, "Classifier has to be a LinearSVM, but is [%s]" \
                              % `type(clf)`

        # init base classes first
        SensitivityAnalyzer.__init__(self)

        self.__clf = clf
        """Classifier that will be trained on datasets and where weights will be
        extracted from."""


    def __call__(self, dataset, callables=[]):
        """Train linear SVM on `dataset` and extract weights from classifier.
        """
        self.__clf.train(dataset)

        # first multiply SV coefficients with the actuall SVs to get weighted
        # impact of SVs on decision, then for each feature take absolute mean
        # across SVs to get a single weight value per feature
        return N.abs((N.matrix(self.__clf.model.getSVCoef())
                      * N.matrix(self.__clf.model.getSV())).mean(axis=0).A1)
