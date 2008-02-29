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

from mvpa.clfs.svm import LinearSVM
from mvpa.algorithms.datameasure import ClassifierBasedSensitivityAnalyzer, \
     selectAnalyzer
from mvpa.misc import warning
from mvpa.misc.state import StateVariable


if __debug__:
    from mvpa.misc import debug

class LinearSVMWeights(ClassifierBasedSensitivityAnalyzer):
    """`SensitivityAnalyzer` that reports the weights of a linear SVM trained
    on a given `Dataset`.
    """

    offsets = StateVariable(enabled=True,
                            doc="Offsets of separating hyperplane")

    def __init__(self, clf, **kwargs):
        """Initialize the analyzer with the classifier it shall use.

        :Parameters:
          clf: LinearSVM
            classifier to use. Only classifiers sub-classed from
            `LinearSVM` may be used.
        """
        if not isinstance(clf, LinearSVM):
            raise ValueError, \
                  "Classifier %s has to be a LinearSVM, but is [%s]" \
                              % (`clf`, `type(clf)`)

        # init base classes first
        ClassifierBasedSensitivityAnalyzer.__init__(self, clf, **kwargs)


    def _call(self, dataset):
        """Extract weights from Linear SVM classifier.
        """
        if self.clf.model.nr_class != 2:
            warning("You are estimating sensitivity for SVM %s trained on %d" %
                    (`self.clf`, self.clf.model.nr_class) +
                    " classes. Make sure that it is what you intended to do" )
        svcoef = N.matrix(self.clf.model.getSVCoef())
        svs = N.matrix(self.clf.model.getSV())
        rhos = N.array(self.clf.model.getRho())

        self.offsets = rhos
        # XXX yoh: .mean() is effectively
        # averages across "sensitivities" of all paired classifiers (I
        # think). See more info on this topic in svm.py on how sv_coefs
        # are stored
        #
        # First multiply SV coefficients with the actuall SVs to get
        # weighted impact of SVs on decision, then for each feature
        # take mean across SVs to get a single weight value
        # per feature
        weights = (svcoef * svs).mean(axis=0).A1

        if __debug__:
            debug('SVM',
                  "Extracting weights for %d-class SVM: #SVs=%s, " %
                  (self.clf.model.nr_class, `self.clf.model.getNSV()`) +
                  " SVcoefshape=%s SVs.shape=%s Rhos=%s. Result: min=%f max=%f" %\
                  (svcoef.shape, svs.shape, rhos, N.min(weights), N.max(weights)))

        return weights

