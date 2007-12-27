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

from mvpa.algorithms.datameasure import ClassifierBasedSensitivityAnalyzer
from mvpa.clfs.svm import LinearSVM


class LinearSVMWeights(ClassifierBasedSensitivityAnalyzer):
    """`SensitivityAnalyzer` that reports the weights of a linear SVM trained
    on a given `Dataset`.
    """
    def __init__(self, clf, **kwargs):
        """Initialize the analyzer with the classifier it shall use.

        :Parameters:
          clf : LinearSVM
            classifier to use. Only classifiers sub-classed from
            `LinearSVM` may be used.
        """
        if not isinstance(clf, LinearSVM):
            raise ValueError, "Classifier has to be a LinearSVM, but is [%s]" \
                              % `type(clf)`

        # init base classes first
        ClassifierBasedSensitivityAnalyzer.__init__(self, clf, **kwargs)


    def _call(self, dataset, callables=[]):
        """Extract weights from Linear SVM classifier.
        """
        # first multiply SV coefficients with the actuall SVs to get weighted
        # impact of SVs on decision, then for each feature take absolute mean
        # across SVs to get a single weight value per feature
        return N.abs((N.matrix(self.clf.model.getSVCoef())
                      * N.matrix(self.clf.model.getSV())).mean(axis=0).A1)
