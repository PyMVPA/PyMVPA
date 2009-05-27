# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Provide sensitivity measures for sg's SVM."""

__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.base import externals
if externals.exists('shogun', raiseException=True):
    import shogun.Classifier

from mvpa.misc.state import StateVariable
from mvpa.measures.base import Sensitivity

if __debug__:
    from mvpa.base import debug


class LinearSVMWeights(Sensitivity):
    """`Sensitivity` that reports the weights of a linear SVM trained
    on a given `Dataset`.
    """

    biases = StateVariable(enabled=True,
                           doc="Offsets of separating hyperplanes")

    def __init__(self, clf, **kwargs):
        """Initialize the analyzer with the classifier it shall use.

        :Parameters:
          clf: LinearSVM
            classifier to use. Only classifiers sub-classed from
            `LinearSVM` may be used.
        """
        # init base classes first
        Sensitivity.__init__(self, clf, **kwargs)


    def __sg_helper(self, svm):
        """Helper function to compute sensitivity for a single given SVM"""
        self.offsets = svm.get_bias()
        svcoef = N.matrix(svm.get_alphas())
        svnums = svm.get_support_vectors()
        svs = self.clf.traindataset.samples[svnums,:]
        res = (svcoef * svs).mean(axis=0).A1
        return res


    def _call(self, dataset):
        # XXX Hm... it might make sense to unify access functions
        # naming across our swig libsvm wrapper and sg access
        # functions for svm
        svm = self.clf.svm
        if isinstance(svm, shogun.Classifier.MultiClassSVM):
            sens = []
            for i in xrange(svm.get_num_svms()):
                sens.append(self.__sg_helper(svm.get_svm(i)))
        else:
            sens = self.__sg_helper(svm)
        return N.asarray(sens)

