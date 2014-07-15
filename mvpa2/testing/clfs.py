# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Provides `clfs` dictionary with instances of all available classifiers."""

__docformat__ = 'restructuredtext'

# Global modules
import numpy as np

# Some global imports useful through out unittests
from mvpa2.base import cfg

# Base classes
from mvpa2.clfs.base import Classifier
from mvpa2.datasets.base import Dataset
from mvpa2.measures.base import FeaturewiseMeasure

#
# first deal with classifiers which do not have external deps
#
from mvpa2.clfs.dummies import *
from mvpa2.clfs.smlr import SMLR
from mvpa2.clfs.knn import *

from mvpa2.clfs.warehouse import clfswh, regrswh
from mvpa2.base import externals


__all__ = ['clfswh', 'regrswh', 'Classifier', 'SameSignClassifier',
           'Less1Classifier', 'sample_clf_nl', 'sample_clf_lin',
           'sample_clf_reg', 'cfg', 'SillySensitivityAnalyzer']

# if have ANY svm implementation
if externals.exists('libsvm') or externals.exists('shogun'):
    from mvpa2.clfs.svm import *
    __all__ += ['LinearCSVMC']
    if externals.exists('libsvm'):
        __all__ += ['libsvm', 'LinearNuSVMC']
    if externals.exists('shogun'):
        __all__ += ['sg']


class SillySensitivityAnalyzer(FeaturewiseMeasure):
    """Simple one which just returns xrange[-N/2, N/2], where N is the
    number of features
    """
    is_trained = True

    def __init__(self, mult=1, **kwargs):
        FeaturewiseMeasure.__init__(self, **kwargs)
        self.__mult = mult

    def _call(self, dataset):
        """Train linear SVM on `dataset` and extract weights from classifier.
        """
        sens = self.__mult *( np.arange(dataset.nfeatures) - int(dataset.nfeatures/2) )
        return Dataset(sens[np.newaxis])



# Sample universal classifiers (linear and non-linear) which should be
# used whenever it doesn't matter what classifier it is for testing
# some higher level creations -- chosen so it is the fastest universal
# one. Also it should not punch state.py in the face how it is
# happening with kNN...
sample_clf_lin = SMLR(lm=0.1)#sg.svm.LinearCSVMC(svm_impl='libsvm')

#if externals.exists('shogun'):
#    sample_clf_nl = sg.SVM(kernel_type='RBF', svm_impl='libsvm')
#else:
#classical one which was used for a while
#and surprisingly it is not bad at all for the unittests
sample_clf_nl = kNN(k=5)

# and also a regression-based classifier
r = clfswh['linear', 'regression_based', 'has_sensitivity']
if len(r) > 0: sample_clf_reg = r[0]
else: sample_clf_reg = None
