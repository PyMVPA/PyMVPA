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
from mvpa.clfs import libsvm, sg
import mvpa.clfs.libsvm.svm
import mvpa.clfs.sg.svm
from mvpa.clfs.smlr import SMLR
from mvpa.clfs.ridge import *
from mvpa.clfs.knn import *


clfs={'LinearSVMC' : [libsvm.svm.LinearCSVMC(probability=1),
                      libsvm.svm.LinearNuSVMC(probability=1), sg.svm.LinearCSVMC()],
      'NonLinearSVMC' : [libsvm.svm.RbfCSVMC(probability=1),
                         libsvm.svm.RbfNuSVMC(probability=1), sg.svm.RbfCSVMC()]
      }

clfs['SVMC'] = clfs['LinearSVMC'] + clfs['NonLinearSVMC']

clfs['LinearC'] = clfs['LinearSVMC'] + \
                  [ SMLR(implementation="Python"), SMLR(implementation="C") ]

clfs['NonLinearC'] = clfs['NonLinearSVMC'] + [ kNN(k=1), RidgeReg() ]

clfs['all'] = clfs['LinearC'] + clfs['NonLinearC']

clfs['clfs_with_sens'] =  clfs['LinearC']

