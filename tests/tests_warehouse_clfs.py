#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Provides `clfs` dictionary with instances of all available classifiers."""

__docformat__ = 'restructuredtext'

#
# first deal with classifiers which do not have external deps
#
from mvpa.clfs.smlr import SMLR
from mvpa.clfs.ridge import RidgeReg
from mvpa.clfs.knn import *

from mvpa.clfs.warehouse import clfs
from mvpa.base import externals

# if have ANY svm implementation
if externals.exists('libsvm') or externals.exists('shogun'):
    from mvpa.clfs.svm import *

#
# Few silly classifiers
#
class SameSignClassifier(Classifier):
    """Dummy classifier which reports +1 class if both features have
    the same sign, -1 otherwise"""

    def __init__(self, **kwargs):
        Classifier.__init__(self, train2predict=False, **kwargs)

    def _train(self, data):
        # we don't need that ;-)
        pass

    def _predict(self, data):
        datalen = len(data)
        values = []
        for d in data:
            values.append(2*int( (d[0]>=0) == (d[1]>=0) )-1)
        self.predictions = values
        return values


class Less1Classifier(SameSignClassifier):
    """Dummy classifier which reports +1 class if abs value of max less than 1"""
    def _predict(self, data):
        datalen = len(data)
        values = []
        for d in data:
            values.append(2*int(max(d)<=1)-1)
        self.predictions = values
        return values

# Sample universal classifiers (linear and non-linear) which should be
# used whenever it doesn't matter what classifier it is for testing
# some higher level creations -- chosen so it is the fastest universal
# one. Also it should not punch state.py in the face how it is
# happening with kNN...
sample_clf_lin = SMLR(lm=0.1)#sg.svm.LinearCSVMC(svm_impl='libsvm')

if externals.exists('shogun'):
    sample_clf_nl = sg.svm.RbfCSVMC(svm_impl='libsvm')
else:
    #classical one which was used for a while
    #and surprisingly it is not bad at all for the unittests
    sample_clf_nl = kNN(k=5)
