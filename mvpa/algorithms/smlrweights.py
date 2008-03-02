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

from mvpa.clfs.svm import SMLR
from mvpa.algorithms.datameasure import ClassifierBasedSensitivityAnalyzer
from mvpa.misc import warning
from mvpa.misc.state import StateVariable


if __debug__:
    from mvpa.misc import debug

class SMLRWeights(ClassifierBasedSensitivityAnalyzer):
    """`SensitivityAnalyzer` that reports the weights SMLR trained
    on a given `Dataset`.
    """

    # SMLR has no such one right?
    #offsets = StateVariable(enabled=True,
    #                        doc="Offsets of separating hyperplane")

    def __init__(self, clf, **kwargs):
        """Initialize the analyzer with the classifier it shall use.

        :Parameters:
          clf: SMLR
            classifier to use. Only classifiers sub-classed from
            `SMLR` may be used.
        """
        if not isinstance(clf, SMLR):
            raise ValueError, \
                  "Classifier %s has to be a SMLR, but is [%s]" \
                              % (`clf`, `type(clf)`)

        # init base classes first
        ClassifierBasedSensitivityAnalyzer.__init__(self, clf, **kwargs)


    def _call(self, dataset):
        """Extract weights from Linear SVM classifier.
        """
        if self.clf.w.shape[1] != 1:
            warning("You are estimating sensitivity for SVM %s trained on %d" %
                    (`self.clf`, self.clf.w.shape[1]+1) +
                    " classes. Make sure that it is what you intended to do" )

        weights = N.mean(self.clf.smlr.w, axis=0)

        if __debug__:
            debug('SVM',
                  "Extracting weights for %d-class SMLR" %
                  (self.clf.w.shape[1]+!) +
                  "Result: min=%f max=%f" %\
                  (N.min(weights), N.max(weights)))

        return weights

