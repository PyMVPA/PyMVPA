#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Miscelaneous functions/datasets to be used in the unit tests"""

__docformat__ = 'restructuredtext'


# Define sets of classifiers
from mvpa.clfs.svm import *
from mvpa.clfs.smlr import SMLR
from mvpa.clfs.ridge import *
from mvpa.clfs.knn import *

clfs={'LinearSVMC' : [LinearCSVMC(), LinearNuSVMC()],
      'NonLinearSVMC' : [RbfCSVMC(), RbfNuSVMC()],
      }

clfs['LinearC'] = clfs['LinearSVMC'] + \
                  [ SMLR(implementation="Python"), SMLR(implementation="C") ]

clfs['NonLinearC'] = clfs['NonLinearSVMC'] + [ kNN(k=1), RidgeReg() ]

clfs['all'] = clfs['LinearC'] + clfs['NonLinearC']

clfs['clfs_with_sens'] =  clfs['LinearC']

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


