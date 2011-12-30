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

import numpy as np

from mvpa2.base import externals
if externals.exists('shogun', raise_=True):
    import shogun.Classifier
    _shogun_exposes_slavesvm_labels = externals.versions['shogun:rev'] < 4633

from mvpa2.base.state import ConditionalAttribute
from mvpa2.base.types import asobjarray
from mvpa2.measures.base import Sensitivity
from mvpa2.datasets.base import Dataset

if __debug__:
    from mvpa2.base import debug


class LinearSVMWeights(Sensitivity):
    """`Sensitivity` that reports the weights of a linear SVM trained
    on a given `Dataset`.
    """

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
        bias = svm.get_bias()
        # if it has get_w (linear ones like SVMOcas) -- use it,
        # otherwise resort to recomputing
        if hasattr(svm, 'get_w'):
            res = svm.get_w()
        else:
            svcoef = np.matrix(svm.get_alphas())
            svnums = svm.get_support_vectors()
            svs = self.clf.traindataset.samples[svnums,:]
            res = (svcoef * svs).mean(axis=0).A1
        return res, bias


    def _call(self, dataset):
        # XXX Hm... it might make sense to unify access functions
        # naming across our swig libsvm wrapper and sg access
        # functions for svm
        clf = self.clf
        sgsvm = clf.svm
        sens_labels = None
        if isinstance(sgsvm, shogun.Classifier.MultiClassSVM):
            sens, biases = [], []
            nsvms = sgsvm.get_num_svms()
            clabels = sorted(clf._attrmap.values())
            nclabels = len(clabels)
            sens_labels = []
            isvm = 0                    # index for svm among known

            for i in xrange(nclabels):
                for j in xrange(i+1, nclabels):
                    sgsvmi = sgsvm.get_svm(isvm)
                    labels_tuple = (clabels[i], clabels[j])
                    # Since we gave the labels in incremental order,
                    # we always should be right - but it does not
                    # hurt to check if set of labels is the same
                    if __debug__ and _shogun_exposes_slavesvm_labels:
                        if not sgsvmi.get_labels():
                            # We need to call classify() so labels get assigned
                            # to the multiclass SVM
                            sgsvm.classify()
                        assert(set([sgsvmi.get_label(int(x))
                                    for x in sgsvmi.get_support_vectors()])
                               == set(labels_tuple))
                    sens1, bias = self.__sg_helper(sgsvmi)
                    sens.append(sens1)
                    biases.append(bias)
                    sens_labels += [labels_tuple[::-1]] # ??? positive first
                    isvm += 1
            assert(len(sens) == nsvms)  # we should have  covered all
        else:
            sens1, bias = self.__sg_helper(sgsvm)
            biases = np.atleast_1d(bias)
            sens = np.atleast_2d(sens1)
            if not clf.__is_regression__:
                assert(set(clf._attrmap.values()) == set([-1.0, 1.0]))
                assert(sens.shape[0] == 1)
                sens_labels = [(-1.0, 1.0)]

        ds = Dataset(np.atleast_2d(sens))
        if sens_labels is not None:
            if isinstance(sens_labels[0], tuple):
                # Need to have them in array of dtype object
                sens_labels = asobjarray(sens_labels)

            if len(clf._attrmap):
                sens_labels = clf._attrmap.to_literal(sens_labels, recurse=True)
            ds.sa[clf.get_space()] = sens_labels
        ds.sa['biases'] = biases

        return ds
