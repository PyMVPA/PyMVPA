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
from mvpa.base import cfg

# Base classes
from mvpa.clfs.base import Classifier
from mvpa.datasets.base import Dataset
from mvpa.measures.base import FeaturewiseMeasure

#
# first deal with classifiers which do not have external deps
#
from mvpa.clfs.base import Classifier
from mvpa.clfs.smlr import SMLR
from mvpa.clfs.knn import *

from mvpa.clfs.warehouse import clfswh, regrswh
from mvpa.base import externals
from mvpa.base.types import accepts_dataset_as_samples

__all__ = ['clfswh', 'regrswh', 'Classifier', 'SameSignClassifier',
           'Less1Classifier', 'sample_clf_nl', 'sample_clf_lin',
           'sample_clf_reg', 'cfg', 'SillySensitivityAnalyzer']

# if have ANY svm implementation
if externals.exists('libsvm') or externals.exists('shogun'):
    from mvpa.clfs.svm import *
    __all__ += ['LinearCSVMC']
    if externals.exists('libsvm'):
        __all__ += ['libsvm', 'LinearNuSVMC']
    if externals.exists('shogun'):
        __all__ += ['sg']
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


class Less1Classifier(SameSignClassifier):
    """Dummy classifier which reports +1 class if abs value of max less than 1"""
    def _predict(self, data):
        datalen = len(data)
        estimates = []
        for d in data:
            estimates.append(2*int(max(d)<=1)-1)
        self.predictions = estimates
        return estimates


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
