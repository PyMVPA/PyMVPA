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
from mvpa.datasets.base import Dataset

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

        Parameters
        ----------
        clf : LinearSVM
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
        clf = self.clf
        sgsvm = clf.svm
        sens_labels = None
        if isinstance(sgsvm, shogun.Classifier.MultiClassSVM):
            sens = []
            nsvms = sgsvm.get_num_svms()
            clabels = sorted(clf._attrmap.values())
            nclabels = len(clabels)
            sens_labels = []
            for i in xrange(nclabels):
                for j in xrange(i+1, nclabels):
                    sens.append(self.__sg_helper(sgsvm.get_svm(i)))
                    sens_labels += [(clabels[i], clabels[j])]
            assert(len(sens) == nsvms)
        else:
            sens = N.atleast_2d(self.__sg_helper(sgsvm))
            if not clf.__is_regression__:
                assert(set(clf._attrmap.values()) == set([-1.0, 1.0]))
                assert(sens.shape[0] == 1)
                sens_labels = [(-1.0, 1.0)]

        ds = Dataset(N.atleast_2d(sens))
        if sens_labels is not None:
            if len(clf._attrmap):
                sens_labels = clf._attrmap.to_literal(sens_labels, recurse=True)
            ds.sa['labels'] = sens_labels

        return ds
