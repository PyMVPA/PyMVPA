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
      'clfs_with_sens' : [LinearCSVMC(), LinearNuSVMC()],
      }

clfs['LinearC'] = clfs['LinearSVMC'] + [ SMLR(implementation="Python") ]
                # + SMLR(implementation="C")
clfs['NonLinearC'] = clfs['NonLinearSVMC'] + [ kNN(k=1), RidgeReg() ]
clfs['all'] = clfs['LinearC'] + clfs['NonLinearC']

